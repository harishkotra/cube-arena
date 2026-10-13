#!/usr/bin/env python3
"""Rubik's cube arena UI server.

Serves a single page with SEVEN sections, one per decision model, all running the
same scrambled cube at the same time in the same view (three per row).

    python server.py            # http://127.0.0.1:8077
    python server.py --port 9000

Routes
    GET  /                 the UI page
    GET  /api/engines      per-engine load status (the page polls this on startup)
    POST /api/run          start a race  {seed, mode, policy, moves, engines}
    GET  /api/run/<id>     current state of a run (the page polls this while running)

Models are loaded once at startup and reused across runs, so the first start
takes a while (GLiNER2.5-Decide alone is ~40 s to load; strands-decider runs as a
separate server you start yourself, see README).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from arena import run as arena_run
from charts import build_scorecard
from cube import facelet_rows, scramble, solved_faces, solved_facelets
from engines import ClefEngine, GlinerEngine, KevEngine, LayaEngine, PhocinaeEngine, StrandsEngine

ROOT = pathlib.Path(__file__).parent
UI = ROOT / "ui" / "index.html"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Order here is the order of the seven sections in the UI.
ENGINE_SPECS = [
    ("laya", "Laya 421M", "laya-mlx · MLX · local", lambda: LayaEngine()),
    ("strands", "Strands Decider 2B", "strands-decider · torch/MPS · local", lambda: StrandsEngine()),
    ("gliner", "GLiNER2.5-Decide 340M", "gliner2 · DeBERTa-v3 · local", lambda: GlinerEngine()),
    ("clef-flash", "Clef-Flash 9B", "clef-flash · Workers AI · cloud", lambda: ClefEngine("clef-flash")),
    ("clef", "Clef 27B", "clef · Workers AI · cloud", lambda: ClefEngine("clef")),
    ("kev", "Kev-0.8B", "kev · local server :8008", lambda: KevEngine()),
    ("phocinae", "Phocinae Largha 150M", "phocinae · local server :8155", lambda: PhocinaeEngine()),
]

# Engines that emit generated text rather than a typed option distribution. The UI
# flags these so their results are not read as like-for-like decision models.
# Empty: all seven registered engines are typed-decision models returning a
# distribution over the 18 moves.
GENERATIVE: set[str] = set()

# matplotlib's pyplot state machine is not thread-safe. The scorecard is built on
# the run thread after a race and on-demand by the /api/chart/ route; without a
# lock those two calls can interleave and corrupt or fail the render.
CHART_LOCK = threading.Lock()


class Hub:
    """Holds the loaded engines and the one active run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.engines: dict[str, dict] = {
            key: {"key": key, "label": label, "runtime": runtime,
                  "status": "loading", "error": None, "engine": None}
            for key, label, runtime, _ in ENGINE_SPECS
        }
        self.run: dict | None = None

    # -- startup ---------------------------------------------------------- #
    def load_all(self) -> None:
        for key, _label, _runtime, build in ENGINE_SPECS:
            try:
                eng = build()
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.engines[key].update(
                        status="unavailable", engine=None,
                        error=f"{type(e).__name__}: {e}",
                    )
                continue
            # A constructed engine can still be unusable (dead local server, bad
            # key), so probe before claiming it is ready.
            try:
                ok, why = eng.probe()
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"probe failed: {type(e).__name__}: {e}"
            with self.lock:
                if ok:
                    self.engines[key].update(status="ready", engine=eng, error=None)
                else:
                    self.engines[key].update(status="unavailable", engine=eng, error=why)

    def engines_view(self) -> list[dict]:
        with self.lock:
            return [
                {**{k: v[k] for k in ("key", "label", "runtime", "status", "error")},
                 "generative": v["key"] in GENERATIVE}
                for v in self.engines.values()
            ]

    # -- runs ------------------------------------------------------------- #
    def start_run(self, cfg: dict) -> dict:
        with self.lock:
            if self.run and self.run["status"] == "running":
                raise RuntimeError("a run is already in progress")
            keys = cfg.get("engines") or list(self.engines.keys())
            cube, seq = scramble(cfg["seed"])
            panels = []
            for k in keys:
                spec = self.engines[k]
                panels.append({
                    "key": k, "label": spec["label"], "runtime": spec["runtime"],
                    "status": "ready" if spec["status"] == "ready" else spec["status"],
                    "error": spec["error"],
                    "generative": k in GENERATIVE,
                    "moves": 0, "facelets": solved_facelets(cube), "faces": solved_faces(cube),
                    "start_facelets": solved_facelets(cube), "start_faces": solved_faces(cube),
                    "median_ms": None, "net": facelet_rows(cube), "move_log": [], "solved": False,
                    "last_confidence": None, "last_latency": None,
                    "hits": 0, "accepted": 0, "neutrals": 0, "corrections": 0,
                    "optimal_distance": None, "solver": None,
                })
            self.run = {
                "id": uuid.uuid4().hex[:12],
                "status": "running",
                "config": cfg,
                "seed": cfg["seed"],
                "scramble": seq,
                "start": {"facelets": solved_facelets(cube), "faces": solved_faces(cube)},
                "panels": panels,
                "started_at": time.time(),
            }
            run = self.run
        threading.Thread(target=self._drive, args=(run,), daemon=True).start()
        return run

    def _drive(self, run: dict) -> None:
        """Run every ready engine in parallel, each in its own thread."""
        threads = []
        for panel in run["panels"]:
            if panel["status"] != "ready":
                continue
            t = threading.Thread(target=self._one_engine, args=(run, panel), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        # Persist the run and render the scorecard BEFORE flipping the status to
        # "done". The page polls /api/run/<id> and requests the chart the moment it
        # sees "done"; if the PNG is not on disk yet the <img> 404s and the
        # scorecard shows as a broken image. Chart building is also serialised
        # (CHART_LOCK) so it cannot race the on-demand /api/chart/ route.
        finished_at = time.time()
        try:
            # Snapshot with the terminal status: the live dict is only flipped to
            # "done" after the chart is rendered (see below), but the record on
            # disk should read as finished.
            snapshot = {**run, "status": "done", "finished_at": finished_at}
            (RESULTS / f"ui_run_{run['id']}.json").write_text(json.dumps(snapshot, indent=2))
        except Exception:  # noqa: BLE001
            pass
        # Best-effort scorecard chart; a chart failure must never break a run.
        try:
            with CHART_LOCK:
                build_scorecard(run, RESULTS / f"chart_{run['id']}.png")
        except Exception:  # noqa: BLE001
            pass
        with self.lock:
            run["status"] = "done"
            run["finished_at"] = finished_at

    def _one_engine(self, run: dict, panel: dict) -> None:
        eng = self.engines[panel["key"]]["engine"]
        panel["status"] = "running"
        lat: list[float] = []

        def on_move(ev: dict) -> None:
            with self.lock:
                if ev.get("applied"):
                    panel["moves"] += 1
                    tag = ev.get("tag")
                    if tag == "hit":
                        panel["hits"] += 1
                    elif tag == "accepted":
                        panel["accepted"] += 1
                    elif tag == "neutral":
                        panel["neutrals"] += 1
                    elif tag == "corrected":
                        panel["corrections"] += 1
                    if ev.get("optimal_distance") is not None:
                        panel["optimal_distance"] = ev["optimal_distance"]
                    if ev.get("solver"):
                        panel["solver"] = ev["solver"]
                    panel["move_log"].append({
                        "i": ev["i"], "move": ev["applied"], "argmax": ev["argmax"],
                        "facelets": ev["facelets"], "confidence": ev["confidence"],
                        "latency_ms": ev["latency_ms"], "tag": ev.get("tag"),
                    })
                    if len(panel["move_log"]) > 200:
                        panel["move_log"] = panel["move_log"][-200:]
                    lat.append(ev["latency_ms"])
                    panel["median_ms"] = sorted(lat)[len(lat) // 2]
                panel["facelets"] = ev["facelets"]
                panel["faces"] = ev["faces"]
                panel["net"] = ev["net"]
                panel["last_confidence"] = ev["confidence"]
                panel["last_latency"] = ev["latency_ms"]
                if ev["faces"] == 6:
                    panel["solved"] = True

        try:
            cfg = run["config"]
            res = arena_run(eng, cfg["seed"], cfg["mode"], cfg["moves"],
                            cfg["policy"], on_move=on_move)
            with self.lock:
                panel["status"] = "done"
                panel["final_facelets"] = res["final_facelets"]
                panel["final_faces"] = res["final_faces"]
                panel["solved"] = res["solved"]
                panel["median_ms"] = res["latency_ms_median"]
                panel["hits"] = res.get("hits", 0)
                panel["accepted"] = res.get("accepted", 0)
                panel["neutrals"] = res.get("neutrals", 0)
                panel["corrections"] = res.get("corrections", 0)
                panel["optimal_distance"] = res.get("optimal_distance")
                panel["solver"] = res.get("solver")
        except Exception as e:  # noqa: BLE001
            with self.lock:
                panel["status"] = "error"
                panel["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    def run_view(self, run_id: str) -> dict | None:
        with self.lock:
            if not self.run or self.run["id"] != run_id:
                return None
            return json.loads(json.dumps(self.run))  # snapshot under lock


HUB = Hub()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not UI.exists():
                return self._send(500, b"ui/index.html missing", "text/plain")
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/engines":
            return self._json(200, {"engines": HUB.engines_view()})
        if path.startswith("/api/chart/"):
            run_id = path.rsplit("/", 1)[-1]
            png = RESULTS / f"chart_{run_id}.png"
            try:
                with CHART_LOCK:
                    if not png.exists():
                        # Allow charting any run that was written to disk, not just
                        # the one currently in memory.
                        run_json = RESULTS / f"ui_run_{run_id}.json"
                        if not run_json.exists():
                            return self._json(404, {"error": "no chart for that run"})
                        build_scorecard(json.loads(run_json.read_text()), png)
                    data = png.read_bytes()
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": f"chart failed: {type(e).__name__}: {e}"})
            return self._send(200, data, "image/png")
        if path.startswith("/api/run/"):
            run_id = path.rsplit("/", 1)[-1]
            view = HUB.run_view(run_id)
            if view is None:
                return self._json(404, {"error": "unknown run"})
            return self._json(200, view)
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/run":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:  # noqa: BLE001
            return self._json(400, {"error": "bad json"})
        cfg = {
            "seed": int(body.get("seed", 20261002)),
            "mode": body.get("mode", "greedy"),
            "policy": body.get("policy", "sample"),
            "moves": int(body.get("moves", 24)),
            "engines": body.get("engines") or None,
        }
        try:
            run = HUB.start_run(cfg)
        except Exception as e:  # noqa: BLE001
            return self._json(409, {"error": str(e)})
        return self._json(200, {"run_id": run["id"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    threading.Thread(target=HUB.load_all, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"cube arena UI -> http://{args.host}:{args.port}  (models loading in background)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
