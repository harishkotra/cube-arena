"""One interface, seven engines.

All seven decision models speak the same contract:

    {state, questions: {id: {type, instructions, criteria}}} -> answers[id]

  * strands-decider  : POST http://127.0.0.1:8010/v1/systemone    (local, torch/MPS)
  * laya-mlx         : in-process agent.predict(state, questions)  (local, MLX)
  * gliner2.5-decide : in-process classify_text(state, labels)     (local, CPU)
  * clef / clef-flash: POST Cloudflare Workers AI /ai/run/...      (cloud)
  * kev              : POST http://127.0.0.1:8008/v1/systemone    (local server)
  * phocinae         : POST http://127.0.0.1:8155/v1/systemone    (local server)

Each adapter returns a Decision: the chosen option, the confidence, the full
probability distribution, and the wall-clock latency of the call.
"""

from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from cube import MOVES


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from `.env` next to this file.

    Real environment variables always win, so exporting still works. This exists so
    the Cloudflare keys (and the optional HF token) can live in ~/cube-arena/.env
    instead of being exported in every shell you start the arena from. No
    third-party dependency.
    """
    path = pathlib.Path(__file__).parent / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


@dataclass
class Decision:
    engine: str
    choice: str
    confidence: float | None
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


def _parse_answers(resp: dict, qid: str) -> tuple[str, float | None, dict[str, float]]:
    """Pull (choice, confidence, distribution) out of an answers[id] payload."""
    ans = resp["answers"][qid]
    if isinstance(ans, str):
        return ans, None, {}
    choice = ans.get("choice") or ans.get("answer") or ans.get("value")
    conf = ans.get("confidence")
    probs = ans.get("probabilities") or ans.get("distribution") or {}
    if isinstance(probs, dict):
        probs = {k: float(v) for k, v in probs.items()}
    return choice, (float(conf) if conf is not None else None), probs


_FACE_NAME = {"U": "up", "R": "right", "F": "front", "D": "down", "L": "left", "B": "back"}

# strands-decider (and the cloud engines) want choice.criteria as {option: description}.
# laya accepts a dict too, so one shape works for all of them.
MOVE_CRITERIA: dict[str, str] = {}
for _f, _n in _FACE_NAME.items():
    MOVE_CRITERIA[_f] = f"turn the {_n} face clockwise 90 degrees"
    MOVE_CRITERIA[_f + "'"] = f"turn the {_n} face anticlockwise 90 degrees"
    MOVE_CRITERIA[_f + "2"] = f"turn the {_n} face 180 degrees"


def move_question(instructions: str) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(MOVE_CRITERIA)}


class Engine:
    name = "engine"

    def decide(self, state: str, question: dict[str, Any], qid: str = "move") -> Decision:
        raise NotImplementedError

    def probe(self) -> tuple[bool, str | None]:
        """Reachability check. In-process engines are always reachable."""
        return True, None

    def close(self) -> None:
        pass


class HttpEngine(Engine):
    """Shared implementation for any /v1/systemone endpoint."""

    def __init__(self, name: str, url: str, model: str, headers: dict[str, str] | None = None,
                 timeout: float = 180.0, health_url: str | None = None):
        self.name = name
        self.url = url
        self.model = model
        self.timeout = timeout
        self.health_url = health_url
        self.client = httpx.Client(timeout=timeout, headers=headers or {})

    def probe(self) -> tuple[bool, str | None]:
        """Check the backing server, so a dead one reads as unavailable rather
        than silently reporting 'ready' and failing on every move."""
        if not self.health_url:
            return True, None
        try:
            r = httpx.get(self.health_url, timeout=3.0)
        except Exception as e:  # noqa: BLE001
            return False, f"no server at {self.health_url} ({type(e).__name__})"
        if r.status_code >= 500:
            return False, f"HTTP {r.status_code} from {self.health_url}"
        return True, None

    def decide(self, state: str, question: dict[str, Any], qid: str = "move") -> Decision:
        body = {"state": state, "model": self.model, "questions": {qid: question}}
        t0 = time.perf_counter()
        try:
            r = self.client.post(self.url, json=body)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(e).__name__}: {e}")
        dt = (time.perf_counter() - t0) * 1000
        try:
            choice, conf, probs = _parse_answers(resp, qid)
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=dt, error=f"parse: {e}")
        return Decision(self.name, choice, conf, probs, dt)

    def close(self) -> None:
        self.client.close()


class StrandsEngine(HttpEngine):
    def __init__(self, port: int = 8010, host: str = "127.0.0.1"):
        super().__init__(
            name="strands-decider",
            url=f"http://{host}:{port}/v1/systemone",
            model="strands-decider",
            health_url=f"http://{host}:{port}/health",
        )


class LayaEngine(Engine):
    name = "laya-mlx"

    def __init__(self, checkpoint: str = "aac6fef/laya-mlx", dtype: str = "float16"):
        import laya_mlx as laya  # imported lazily; heavy

        self._laya = laya
        self.agent = laya.load(checkpoint, dtype=dtype)

    def decide(self, state: str, question: dict[str, Any], qid: str = "move") -> Decision:
        t0 = time.perf_counter()
        try:
            resp = self.agent.predict(state, {qid: question})
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(e).__name__}: {e}")
        dt = (time.perf_counter() - t0) * 1000
        try:
            choice, conf, probs = _parse_answers(resp, qid)
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=dt, error=f"parse: {e}")
        return Decision(self.name, choice, conf, probs, dt)


class GlinerEngine(Engine):
    """Fastino GLiNER2.5-Decide: an open-weight (Apache 2.0) typed-decision model.

    DeBERTa-v3-large encoder, 340M params, runs locally. `classify_text` takes the
    state and a label set and returns one label (argmax) plus a confidence. It does
    not expose a full distribution, so this engine is argmax-only: the arena's
    sampling policy falls back to its top choice.
    """

    name = "gliner2.5-decide"

    def __init__(self, model_id: str = "fastino/GLiNER2.5-Decide", **kw):
        from gliner2 import AutoExtractor  # lazy; heavy

        self.model = AutoExtractor.from_pretrained(model_id, **kw)

    def decide(self, state: str, question: dict[str, Any], qid: str = "move") -> Decision:
        crit = question["criteria"]
        opts = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        t0 = time.perf_counter()
        try:
            out = self.model.classify_text(state, {qid: opts}, include_confidence=True)
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(e).__name__}: {e}")
        dt = (time.perf_counter() - t0) * 1000
        v = out.get(qid)
        if isinstance(v, dict):
            return Decision(self.name, str(v.get("label") or ""), v.get("confidence"), {}, dt)
        if isinstance(v, (list, tuple)):
            return Decision(self.name, str(v[0]) if v else "", v[1] if len(v) > 1 else None, {}, dt)
        return Decision(self.name, str(v or ""), None, {}, dt)


# ---------------------------------------------------------------------------
# Cloudflare Clef / Clef-Flash
# ---------------------------------------------------------------------------

class ClefEngine(HttpEngine):
    """Cloudflare Clef on Workers AI (`@cf/cloudflare/clef`, `clef-flash`).

    Clef (27B) and Clef-Flash (9B) are decision models that speak the same
    SystemOne contract as the local engines, so this is a drop-in HttpEngine. They run
    on Cloudflare's GPUs, which is the point: a 9B to 27B local model does not fit
    in this machine's 16 GB, but the hosted endpoint costs no local RAM.

    Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN (scope: Workers AI Read).
    """

    def __init__(self, model: str = "clef-flash", account_id: str | None = None,
                 token: str | None = None):
        if model not in ("clef", "clef-flash"):
            raise ValueError(f"unknown clef model: {model!r}")
        acct = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        tok = token or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        if not acct or not tok:
            raise RuntimeError(
                "Clef needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN "
                "(scope 'Workers AI: Read'). Add them to .env."
            )
        super().__init__(
            name=model,
            url=(f"https://api.cloudflare.com/client/v4/accounts/{acct}"
                 f"/ai/run/@cf/cloudflare/{model}"),
            model=model,
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        )

    def decide(self, state: str, question: dict[str, Any], qid: str = "move") -> Decision:
        body = {"state": state, "model": self.model, "questions": {qid: question}}
        t0 = time.perf_counter()
        try:
            r = self.client.post(self.url, json=body)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(e).__name__}: {e}")
        dt = (time.perf_counter() - t0) * 1000
        # Workers AI wraps the SystemOne body in {"result": {...}}; unwrap if present.
        if isinstance(resp, dict) and "answers" not in resp and isinstance(resp.get("result"), dict):
            resp = resp["result"]
        try:
            choice, conf, probs = _parse_answers(resp, qid)
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=dt, error=f"parse: {e}")
        return Decision(self.name, choice, conf, probs, dt)


# ---------------------------------------------------------------------------
# Kev (jaredpalmer/kev)
# ---------------------------------------------------------------------------

class KevEngine(HttpEngine):
    """Kev: a small Jev-style decision model with a local SystemOne server.

        uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008

    The server binds 127.0.0.1 with no auth, and the contract matches, so this
    is another drop-in HttpEngine. On a 16 GB Apple Silicon machine only the
    0.8B checkpoint fits; the family also ships 4B, 9B and 27B, which want a
    32 GB+ machine. The server also exposes GET /v1/models, used as the probe.
    """

    def __init__(self, port: int = 8008, host: str = "127.0.0.1", model: str = "kev-latest",
                 name: str = "kev-0.8b"):
        super().__init__(
            name=name,
            url=f"http://{host}:{port}/v1/systemone",
            model=model,
            headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
            health_url=f"http://{host}:{port}/v1/models",
        )


class PhocinaeEngine(HttpEngine):
    """Phocinae-Largha-150M-v1: a small bilingual decision model with its own
    SystemOne-compatible server.

        PHOC_MODEL_DIR=/path/to/hf_repo python -m phocinae.main

    The server binds 127.0.0.1:8155 by default, runs on CPU or CUDA, and needs
    no auth unless PHOC_BEARER_TOKEN is set. Two shape differences from the
    other engines are handled here: the request carries `questions` as a list
    (`id` / `type` / `options`) rather than a dict, and a choice answer comes
    back as the 0-based option index with per-option probabilities in the
    `option_scores` extension, not as an option key inside `answers`. This
    adapter translates both, so the arena still sees the usual key-based
    contract; the question's `instructions` text has no counterpart in the
    server request and is dropped. GET /health doubles as the probe.
    """

    def __init__(self, port: int = 8155, host: str = "127.0.0.1",
                 model: str = "Phocinae-Largha-150M-v1", name: str = "phocinae",
                 headers: dict[str, str] | None = None):
        super().__init__(
            name=name,
            url=f"http://{host}:{port}/v1/systemone",
            model=model,
            headers=headers,
            health_url=f"http://{host}:{port}/health",
        )

    def decide(self, state: str, question: dict[str, Any], qid: str = "move") -> Decision:
        criteria = question.get("criteria") or {}
        keys = list(criteria.keys())
        if not keys:
            return Decision(self.name, "", None,
                            error="phocinae adapter expects a choice question with criteria")
        body = {
            "model": self.model,
            "state": state,
            "questions": [{"id": qid, "type": question.get("type", "choice"),
                           "options": [str(v) for v in criteria.values()]}],
        }
        t0 = time.perf_counter()
        try:
            r = self.client.post(self.url, json=body)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(e).__name__}: {e}")
        dt = (time.perf_counter() - t0) * 1000
        try:
            idx = resp["answers"][qid]
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(keys):
                raise ValueError(f"answer {idx!r} is not an option index")
            scores = (resp.get("option_scores") or {}).get(qid) or []
            probs = {k: float(s) for k, s in zip(keys, scores)}
            conf = (resp.get("answer_confidence") or {}).get(qid)
        except Exception as e:  # noqa: BLE001
            return Decision(self.name, "", None, latency_ms=dt, error=f"parse: {e}")
        return Decision(self.name, keys[idx], (float(conf) if conf is not None else None), probs, dt)
