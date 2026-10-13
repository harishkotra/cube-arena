# Rubik's Cube Decision Arena

Seven decision models, one scrambled Rubik's cube, one screen. Each model is asked the same question every turn — *which single move reduces the number of misplaced stickers the most?* — and picks one move from the 18 legal moves. The UI shows all seven running **side by side in the same view**, each with a real 3D cube updating live.

All seven are typed **decision models** ("system-one" models) sharing one contract. A model that cannot run says so plainly, with the reason, instead of pretending.

The point is not "who wins". A cube is a **planning** problem (system-two: lookahead, search). These are **system-one** models: fast, calibrated judgments about the current state. The arena measures exactly where that breaks down.

#### Screenshots

![Cube Arena 1](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/n8j77b1mi7vx9ce2eeyo.png)

![Cube Arena 2](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/7x42kzffgncx1krrpyav.png)

![Cube Arena 3](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/4ln28i16idil87h8kw1d.png)

![Cube Arena 4](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/cm631jhc75rd9gben9r1.png)

#### Watch

1. [https://youtu.be/zHGsbg92S2k](https://youtu.be/zHGsbg92S2k)
2. [https://youtu.be/pCTgKPE-mo4](https://youtu.be/pCTgKPE-mo4)

---

## The seven models

| # | Section | Model | Params | Where it runs | How it is called |
|---|---|---|---|---|---|
| 1 | `laya` | **Laya** (`aac6fef/laya-mlx`) | 421M | local, Apple silicon | in-process MLX, `laya_mlx` |
| 2 | `strands` | **Strands Decider 2B** (`StrandsAgents/strands-decider-2B-hobson-v19`) | 1.9B | local, torch/MPS | HTTP `POST /v1/systemone` |
| 3 | `gliner` | **GLiNER2.5-Decide** (`fastino/GLiNER2.5-Decide`) | 340M | local, CPU | in-process, `gliner2` |
| 4 | `clef-flash` | **Clef-Flash** (`Cloudflare/clef-flash`) | 9B | Cloudflare Workers AI | HTTP `.../ai/run/@cf/cloudflare/clef-flash` |
| 5 | `clef` | **Clef** (`Cloudflare/clef`) | 27B | Cloudflare Workers AI | HTTP `.../ai/run/@cf/cloudflare/clef` |
| 6 | `kev` | **Kev-0.8B** (`jaredpalmer/kev-0.8b`) | 0.8B | local server `:8008` | HTTP `POST /v1/systemone` |

All seven share the same typed-decision contract: state plus a closed set of options in, a choice and a confidence out. That is what makes a fair side-by-side possible.

### Why two of them run in the cloud

This was built on an **Apple M2 Mac mini with 8 GB RAM**. Clef (27B) does not fit in 8 GB at any quantization, and the Kev sizes that would matter here (4B, 9B) want ~32 GB for bf16. So Clef and Clef-Flash are called on **Cloudflare Workers AI**, where they cost no local RAM. Full analysis in `REPORT.md`.

### Distributions differ, and the arena exposes it

- **laya, strands, clef, clef-flash, kev and phocinae return a full distribution** over the 18 moves, so the arena runs them as a *stochastic policy* (draw the move from the model's own distribution). Argmax collapses: models pick one move every turn.
- **GLiNER2.5-Decide returns only a single label** plus a confidence, so it is argmax-only. In practice it picks `U` on nearly every turn.

---

## Requirements

- macOS on Apple silicon (developed on macOS 15.5, M-series). 16 GB RAM runs the three local models plus Kev-0.8B and Phocinae-Largha comfortably; the Clef panels only need a Cloudflare token.
- Python 3.12.
- `uv` (used here) or plain `pip`.
- ~9 GB free disk (torch, the Qwen3.5-2B base weights, and the local models).
- A Cloudflare account with a Workers AI token **only** for the Clef panels.

---

## Setup

### 1. Create the environment

```bash
mkdir -p ~/cube-arena && cd ~/cube-arena
uv venv --python 3.12
source .venv/bin/activate

uv pip install laya-mlx strands-decider gliner2 pycuber rich matplotlib requests httpx
uv pip install kociemba   # separate: compiles a C extension, see SETUP.md if it fails
```

`kociemba` has no Apple-silicon wheel on PyPI, so it always compiles a small C extension. Keep it in its own command: if the build fails, `uv pip install` aborts the whole command and you would lose `pycuber` too. Without it, `greedy` and `staged` work and only `guided` is unavailable (it says so plainly). Full guide including the fix for a broken Xcode Command Line Tools: `SETUP.md`.

(With plain pip: `python3.12 -m venv .venv && source .venv/bin/activate && pip install ...`.)

### 2. Start the strands-decider server (section 2)

Strands Decider is served over HTTP, so it runs as its own process. Start it once and leave it running. It downloads the Qwen3.5-2B base (~4.5 GB) on first run.

```bash
cd ~/cube-arena && source .venv/bin/activate
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 \
  --device mps --port 8010 --model-name strands-decider
```

Confirm it is up:

```bash
curl -s http://127.0.0.1:8010/health
# {"status":"ok","model":"strands-decider",...,"device":"mps",...}
```

### 3. Credentials (all optional)

Copy the template and fill in whatever you have:

```bash
cd ~/cube-arena
cp .env.example .env
```

The app reads `~/cube-arena/.env` (the directory holding `server.py`). A real environment variable of the same name always wins over the file, so `export ...` also works. **Any model without credentials shows `unavailable` with the exact reason** and the race runs without it. The arena never invents a result.

| key | lights up | where to get it |
|---|---|---|
| `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_AUTH_TOKEN` | `clef`, `clef-flash` | Cloudflare dashboard → API Tokens, scope **Workers AI: Read** |
| `HF_TOKEN` | faster downloads | optional, Hugging Face settings |

### 3b. Kev-0.8B (section 6, optional)

Kev serves its own SystemOne endpoint, so it needs its own process. It also needs its own checkout + venv (it pins `torch<2.9`, which conflicts with the arena's):

```bash
# one-time
git clone https://github.com/jaredpalmer/kev.git ~/experiments/kev
cd ~/experiments/kev && uv sync --extra serve
# every run (or just ./run_kev.sh from the arena)
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
```

On a 16 GB machine, **Kev-0.8B is the only size that fits**: the 4B and 9B checkpoints want ~32 GB for bf16, and `kev.serve` has no built-in quantization knob (the MLX path loads the stored bf16 weights). The arena labels the panel `Kev-0.8B` so the comparison stays honest. If the server is not running, the panel reads `unavailable: no server at 127.0.0.1:8008`.

### 3c. Phocinae-Largha-150M-v1 (section 7, optional)

Phocinae-Largha is a 144.3M bilingual decision model served by its own SystemOne-compatible server on `:8155`; it runs on CPU or CUDA, no GPU required:

```bash
# one-time
git clone https://github.com/Phocinae/phocinae-server.git ~/experiments/phocinae-server
cd ~/experiments/phocinae-server && pip install fastapi uvicorn torch
# weights: https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1 (point PHOC_MODEL_DIR at the weights checkout)
# every run (or just ./run_phocinae.sh from the arena)
PHOC_MODEL_DIR=/path/to/Phocinae-Largha-150M-v1 python -m phocinae.main
```

If the server is not running, the panel reads `unavailable: no server at 127.0.0.1:8155`.

See `SETUP.md` for the full end-to-end install on a fresh machine.

### 4. Pre-download the local weights (optional, saves time later)

The arena downloads them on first load automatically; pre-fetching just moves the wait out of the demo:

```bash
cd ~/cube-arena && source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download as d; d('aac6fef/laya-mlx')"
python -c "from gliner2 import AutoExtractor; AutoExtractor.from_pretrained('fastino/GLiNER2.5-Decide')"
```

---

## Run the UI (the seven-section view)

```bash
cd ~/cube-arena && source .venv/bin/activate
./run.sh                         # starts the strands server AND the UI
# or, by hand:
python server.py                 # serves http://127.0.0.1:8077
python server.py --port 9000     # or pick a port
```

Open **http://127.0.0.1:8077** in a browser.

What you get:

- **Seven sections, one per model** — every engine gets its own card. Each section shows:
  - the model name and where it runs,
  - a **real 3D cube** (CSS 3D transforms: all six faces positioned in 3D space with perspective, slowly rotating, colours updating after every move),
  - a progress bar for correct stickers (out of 54),
  - running stats: moves, correct stickers, complete faces, median ms/move, last confidence,
  - a **move log** of the last moves it chose, each with the sticker count after it.
- A **shared scramble** line at the top (seed + the 25-move scramble), so it is obvious all four start from the identical cube.
- Controls: **seed**, **mode**, **policy**, **moves**, and a **Start race** button.

On startup the page shows `loading models…` per section while the models load (GLiNER2.5-Decide alone takes ~40 s warm, several minutes cold). When at least one section is `ready`, **Start race** enables. Press it: all ready models run **in parallel**, each in its own section, and the cubes update live until every model has made its moves.

### Controls explained

| Control | Values | Meaning |
|---|---|---|
| **seed** | any integer | Fixes the scramble. Same seed = same starting cube for everyone. |
| **mode** | `guided` | **Solver-guided.** A Kociemba solver knows the optimal move; the model picks; if its pick is worse, the solver plays the optimal move instead and the turn is logged `corrected`. The cube always gets solved. |
| | `greedy` | Ask only "which move reduces misplaced stickers most". The unaided baseline. |
| | `staged` | Walk the beginner-method ladder (white cross → first layer → two layers → solved), naming the current subgoal each turn. |
| **policy** | `sample` | Draw the move from the model's returned distribution (stochastic; the honest way to run a decision model in a loop). |
| | `argmax` | Always take the model's top choice (demonstrates the collapse). |
| **moves** | integer | Move cap per model. **Guided mode needs ≥ 28** for this scramble (optimal 22 + up to 6 neutral turns); 24 cuts the slower models off unsolved. Use **32–40** for a clean demo. GLiNER2.5-Decide is slowest (~3–4 s/move), so it sets the pace. |

---

## Tech stack

| Layer | Choice | Why it was picked |
|---|---|---|
| Cube simulation | `pycuber` | pure-Python 3×3; `perform_algo`, `get_face`, `copy` — no native build |
| Optimal search | `kociemba` | two-phase solver; returns a concrete plan and its first move |
| Local runtimes | MLX (`laya`), torch/MPS (`strands`), transformers/CPU (`gliner2`) | all run on Apple silicon with no CUDA |
| Cloud inference | Cloudflare Workers AI | lets a 9B and a 27B model run without local RAM |
| HTTP client | `httpx` | one pooled client per server-backed engine |
| Web server | Python **stdlib** `http.server` + `ThreadingHTTPServer` | zero framework deps, one file, easy to read |
| UI | single-file HTML + **CSS 3D transforms** + vanilla JS | no build step, no bundler, opens instantly |
| Charts | `matplotlib` (`Agg` backend) | the four-panel dark scorecard PNG |
| Terminal summary | `rich` | the CLI summary table |
| Browser automation | Playwright | screenshots and end-to-end UI verification |
| Env / packaging | `uv` + Python 3.12 | fast, reproducible installs |

---

## Architecture

```
                          ┌──────────────────────────────────┐
                          │  browser   http://127.0.0.1:8077  │
                          │  ui/index.html  (CSS-3D cubes)    │
                          └───────────────┬──────────────────┘
        GET /            GET /api/engines │ POST /api/run
        GET /api/run/<id>  GET /api/chart/<id>
                                          │
                          ┌───────────────▼──────────────────┐
                          │  server.py   (stdlib HTTP)        │
                          │  Hub: loaded engines + active run │
                          │  CHART_LOCK serialises matplotlib │
                          └───────┬───────────────────┬───────┘
                                  │                   │
             one thread per engine (run in parallel)  │
                                  ▼                   ▼
                    ┌────────────────────┐   ┌─────────────────────────────┐
                    │ arena.run()        │   │ charts.build_scorecard()    │
                    │  guided | greedy   │   │ -> results/chart_<id>.png   │
                    │  | staged          │   │ progress · latency ·        │
                    └─────────┬──────────┘   │ guided · final              │
                              │              └─────────────────────────────┘
              each turn: decide(state, question)
                              ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  engines.py — one contract, seven adapters                      │
   │   laya      MLX, in-process          gliner   CPU, in-process   │
   │   strands   HTTP 127.0.0.1:8010      kev      HTTP 127.0.0.1:8008
   │   phocinae  HTTP 127.0.0.1:8155                                 │
   │   clef / clef-flash   Cloudflare Workers AI (HTTPS)             │
   └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  cube.py  (pycuber state + metrics)                             │
   │  solver.py (kociemba plan / first move)                         │
   └────────────────────────────────────────────────────────────────┘
```

**Request flow:** the page loads `ui/index.html`, polls `/api/engines` until the models finish loading, then `POST /api/run` starts a race. The server spawns one thread per ready engine, and the page polls `/api/run/<id>` every 350 ms to redraw the live cubes. When every thread joins, the server writes the run JSON and the scorecard PNG **before** flipping the status to `done`, so the chart never 404s.

---

## The one contract every model speaks

Every engine — local or cloud — is a `choice` call over the same closed set of 18 moves. `engines.py` normalises all seven behind a single method:

```python
# engines.py — the interface (one method) and the shape it returns
def decide(self, state: str, question: dict, qid: str = "move") -> Decision:
    ...

@dataclass
class Decision:
    engine: str
    choice: str                      # e.g. "R2"
    confidence: float | None
    probabilities: dict[str, float]  # over the 18 moves (may be empty)
    latency_ms: float
    error: str | None = None
```

```python
# engines.py — the question: a closed choice set, same for every engine
MOVE_CRITERIA = {}                   # {"U": "turn the up face clockwise 90 degrees", ...}
def move_question(instructions: str) -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": dict(MOVE_CRITERIA)}
```

Because the shape is identical, a new model is a ~15-line adapter, not a rewrite.

---

## How a guided turn works

`arena.run_guided` holds a concrete Kociemba plan and scores the model's pick against it every turn:

```python
# arena.py (abridged)
plan = solution(cube)                 # a known-good move list
correct = plan[0]                     # the solver's optimal next move
d = engine.decide(state_text(cube), move_question(GREEDY_Q))
mv = pick_move(d, policy, rng)        # sample from the distribution, or argmax

if mv == correct:
    tag = "hit"                       # matched the solver exactly
elif mv in MOVES:
    probe = cube.copy(); apply_move(probe, mv)
    d_m = len(solution(probe))
    if d_m < len(plan):
        play, tag, replan = mv, "accepted", True      # helped
    elif d_m == len(plan) and neutral_budget > 0:
        play, tag, replan = mv, "neutral", True       # wasted turn
        neutral_budget -= 1
    else:
        play, tag = correct, "corrected"              # solver overrides
else:
    play, tag = correct, "corrected"                  # unusable answer

apply_move(cube, play)
plan = solution(cube) if replan else plan[1:]
```

`hit` / `accepted` / `neutral` / `corrected` are the four things the scorecard's third panel plots. The cube is guaranteed to advance because the solver always holds a valid plan — but **every turn, including a corrected one, costs one of the `moves` you set**, which is why the cap must be ≥ `optimal + 6`.

---

## Guided mode: how the cube actually gets solved

The unaided models cannot solve a cube. Measured, not assumed: no net progress over 24 moves, and the rank correlation between their confidence and the true quality of a move is ~0 or negative. A cube is a planning problem; these are decision models.

So in `guided` mode the **solver is the search and the model is the decision layer inside it**. Each turn:

1. The harness holds a concrete, known-good solution to the current state (Kociemba) and knows the optimal next move.
2. The model is asked the **same 18-move question** as the unaided race, so the two are directly comparable.
3. The harness scores the model's pick:

| Outcome | Meaning |
|---|---|
| `hit` | the model picked exactly the solver's next move |
| `accepted` | the model picked a different move that still made strict progress |
| `neutral` | the model's move did not change the distance (a wasted turn, budgeted) |
| `corrected` | the model's move was worse (or unusable), so the **solver played the optimal move instead** |

The cube is **guaranteed to finish**: the solver always holds a valid plan, and after the neutral budget is spent every move must strictly reduce the distance.

What the video shows, honestly: the cube solves every time, and the models get scored on how often they picked the solver's move. In practice they almost never do — which is the point. The solver is doing the solving; the model is being measured.

---

## Run the CLI (no UI)

The same harness drives headless runs that write JSON to `results/`.

```bash
cd ~/cube-arena && source .venv/bin/activate

# Solver-guided: the cube actually gets solved (any subset of: laya, strands, gliner, clef, clef-flash, kev)
python arena.py --engines laya,strands,gliner --mode guided --policy sample --moves 80

# The unaided baseline (nobody solves it)
python arena.py --engines laya,strands,gliner --mode greedy --policy sample --moves 24

# The beginner-method variant
python arena.py --engines laya,strands,gliner --mode staged --policy sample --moves 60

# Does the model's probability rank moves by true quality? (multi-state probe)
python probe.py --engines laya,strands --states 25

# Summary table + results/progress.png chart
python render.py
```

---

## Files

| File | What it is |
|---|---|
| `cube.py` | 3×3 cube simulation + metrics (correct stickers, complete faces, staged subgoals). Built on `pycuber`. |
| `engines.py` | The seven adapters behind one `decide(state, question)` interface, plus a reachability probe for the server-backed ones. |
| `solver.py` | Kociemba bridge: solution + first move. Holds the pycuber↔Kociemba facelet encoding. Degrades cleanly if kociemba is missing. |
| `charts.py` | Four-panel dark scorecard (progress, latency, guided resolution, final score) rendered from a run JSON. Served at `/api/chart/<run_id>`. |
| `SETUP.md` | Step-by-step install and run guide for a fresh machine, including keys. |
| `arena.py` | The race loop (guided + greedy + staged, sample + argmax), with a live progress callback. |
| `capture_pw.py` | Screenshots the live UI with Playwright (headless Chromium, which can reach localhost). |
| `probe.py` | Measures whether model probabilities correlate with true move quality. |
| `render.py` | Terminal summary table + progress chart. |
| `server.py` | The UI server (stdlib only): `/`, `/api/engines`, `/api/run`, `/api/run/<id>`. |
| `ui/index.html` | The seven-section page. Single file, no build step. |
| `check_run.py` | Poll a UI run from the terminal and print its final state. |
| `REPORT.md` | The written findings and caveats. |
| `BLOG.md` | The long-form technical write-up (architecture, code walkthrough, lessons). |
| `MARKETING.md` | Launch assets: 30s video script, X thread, LinkedIn post. |
| `run.sh` / `run_kev.sh` | One-command launchers for the UI + strands server and the Kev server. |
| `results/` | Run JSON, probe JSON, and charts (gitignored; regenerated at runtime). |

---

## What the runs actually show

Short version (details and caveats in `REPORT.md`).

**The headline: after 24 moves each, all three local models land on exactly
13/54 — the score they started from.** Zero net progress, and no face completed.
Same scramble, seed `20261002`:

| model | moves | start /54 | final /54 | faces /6 | median ms/move |
|---|---|---|---|---|---|
| laya-mlx | 24 | 13 | **13** | 0 | 101 |
| strands-decider | 24 | 13 | **13** | 0 | 1271 |
| GLiNER2.5-Decide | 24 | 13 | **13** | 0 | 3101 |

- **Nobody solves it, and no face is ever completed.** Even 60 moves of subgoal-named play (staged mode) tops out at 17 of 54 stickers.
- **Staging does not help.** Naming the beginner-method subgoal changes nothing.
- **The confidence ordering is not usable.** Across 25 states the rank correlation between a model's probability and the true quality of the move is ~0 or negative. Chance of picking an optimal move is 8.9%; a helping move, 34.4%.
- **GLiNER2.5-Decide collapses hardest** because it returns a single label with no distribution, so it cannot even be sampled: it plays `U` almost every turn.

### Guided mode: the cube solves, and the models still miss

Same scramble, seed `20261002`, optimal distance **22**. Every model solves:

| model | solved | moves | hits | accepted | neutral | solver corrected | median ms/move | wall |
|---|---|---|---|---|---|---|---|---|
| laya-mlx | yes | 28 | 0 | 1 | 6 | 21 | 102 | 4.9 s |
| strands-decider | yes | 28 | 0 | 2 | 6 | 20 | 1276 | 64.2 s |
| GLiNER2.5-Decide | yes | 27 | 4 | 0 | 5 | 18 | 3194 | 95.0 s |

Read it honestly:

- **The solver solves the cube, not the model.** 18 to 21 of every 27 to 28 moves were the solver overriding the model. laya and strands never once picked the solver's move.
- **GLiNER2.5-Decide's 4 hits are an artefact, not skill.** It plays `U` almost every turn, and the optimal plan happened to start with `U` four times.
- **Wall clock is the real race.** Same solve, but laya finishes in ~5 s against strands' 64 s and GLiNER's 95 s.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to build kociemba`, `ld: ... error: unknown architecture` | Broken Xcode Command Line Tools, not a project problem. `sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`, then `uv pip install kociemba`. |
| `ModuleNotFoundError: No module named 'pycuber'` | An earlier install aborted part-way on the kociemba build. Re-install without kociemba first. |
| `command not found: uv` in a background/other shell | `uv` is not on that shell's PATH. Use the full path (`/Users/<you>/.hermes/bin/uv`) or activate the venv first. |
| strands section shows `unavailable` | The strands-decider server is not running. Start it (Setup step 2) and reload. |
| clef sections show `unavailable` | No `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_AUTH_TOKEN`. Add them (Setup step 3) and restart `server.py`. |
| kev section shows `unavailable` | No server on `:8008`. Start it with `./run_kev.sh` (or Setup step 3b), or ignore it. |
| phocinae section shows `unavailable` | No server on `:8155`. Start it with `./run_phocinae.sh` (or Setup step 3c), or ignore it. |
| First `Start race` feels stuck | GLiNER2.5-Decide takes ~3–4 s per move; the whole race waits on the slowest model. Lower **moves** or run fewer models. |
| `422 Unprocessable Entity` from strands | `choice.criteria` must be a **dict** `{option: description}`, not a list. `engines.py` already does this. |
| `AttributeError: 'Cube' object has no attribute 'faces'` | pycuber uses `cube.get_face("U")`, not `cube.faces`. |
| `deepcopy` fails on the cube | Use `cube.copy()` (pycuber's own), not `copy.deepcopy`. |
| laya prints a temperature-clamping `RuntimeWarning` | Expected. Its `choice:11+` bucket ships an out-of-range temperature, clamped at load; treat its confidence as uncalibrated. |

---

## Caveats before you quote any number

- **Parallel runs share the GPU.** In the UI, laya (MLX) and strands (torch/MPS) run at the same time, so their per-move latency is higher than a solo run (strands measured ~1.27 s solo vs ~2.8 s under contention). The CLI runs sequentially and is the better source for latency claims.
- **GLiNER2.5-Decide is argmax-only** (no distribution), so `policy=sample` does not apply to it.
- **Small sample.** One scramble seed, 25 probe states. Enough to separate "no signal" from "strong signal", not for fine-grained comparisons between models.
- **"Optimal" is tie-inflated.** When several moves share the best delta, a random pick hits one 8.9% of the time, not 1/18.
- **Correct stickers is not monotonic under guided solving.** The solver optimises move count, not sticker count, so the `correct /54` line can dip mid-solve (it went 13 → 11 → 17 → 27 → 54 on the laya run). Judge a guided run by `solved`, not by the sticker line.
- **`optimal` means the solver's own solution length**, not a proven minimum. Kociemba's two-phase search is near-optimal, not guaranteed optimal. That is why guided runs take `optimal + a few` moves rather than exactly `optimal`.

---

## Fork & contribute

There is no framework and no build step, so contributing is deliberately easy.

```bash
# 1. fork on GitHub, then clone your fork
git clone https://github.com/harishkotra/cube-arena.git && cd cube-arena

# 2. make an isolated env (Python 3.12)
uv venv --python 3.12 && source .venv/bin/activate

# 3. install deps (kociemba separately — it compiles a C extension)
uv pip install laya-mlx strands-decider gliner2 pycuber rich matplotlib requests httpx
uv pip install kociemba

# 4. run it
python server.py        # http://127.0.0.1:8077

# 5. verify a change end to end
python verify_ui.py             # checks the page + a real run
python test_server_guided.py    # guided-mode server test
python probe.py --engines laya --states 5
```

Guidelines:

- **Keep the contract.** A new engine must return a `Decision` over the 18 moves and must be reachable via the same `decide(state, question)` call. No engine gets extra context, retries, or a better prompt — that is the whole point of the arena.
- **Never fake a result.** If a model or server is missing, its panel must read `unavailable` with the reason. No synthetic scores, no placeholders.
- **Keep it honest in the UI.** New metrics go next to the existing ones; nothing should imply a model solved a cube the solver solved.
- **Small, single-purpose files.** The server is stdlib-only on purpose; keep it that way unless there is a strong reason.

### Adding a new engine (the common contribution)

If the model exposes an HTTP SystemOne endpoint, subclass `HttpEngine`:

```python
# engines.py
class MyEngine(HttpEngine):
    def __init__(self, port: int = 9001):
        super().__init__(
            name="my-model",
            url=f"http://127.0.0.1:{port}/v1/systemone",
            model="my-model",
            health_url=f"http://127.0.0.1:{port}/v1/models",   # used by probe()
        )
```

Then register it in `server.py`:

```python
ENGINE_SPECS = [
    # ...existing seven...
    ("my-model", "My Model 1B", "my-runtime · local", lambda: MyEngine()),
]
```

If it runs in-process, subclass `Engine` directly and implement `decide()`. Either
way the UI, the charts, and the CLI pick it up with no other changes.

---

## Feature ideas (good first contributions)

- **New modes** — a `reverse` mode (solver picks the *worst* legal move to test recovery), or a `budgeted` mode where the model may spend a limited number of "undos".
- **More metrics** — per-move regret (how much worse than optimal the pick was), a calibration curve (confidence vs. correctness), and a Brier score per model.
- **Best-of-N** — let each model draw N moves and take the best by true distance, to separate "bad distribution" from "bad single draw".
- **Adapters** — add any OpenAI-compatible or llama.cpp server as an engine (the HTTP path is already generic).
- **Run history** — a `/api/runs` index plus a dropdown to reload any past run via the existing `?run=<id>` deep link.
- **Export** — CSV/JSON download of a run's move log, and an MP4/GIF recorder for the animated cubes.
- **Multi-scramble suite** — run the same models across 50 seeds and chart the aggregate, instead of one scramble.
- **Tests** — pytest coverage for `cube.py` metrics, `solver.py` encoding, and the guided state machine; a CI job that runs `verify_ui.py` headless.
- **Layout toggle** — a 2-col / 3-col switch for large monitors vs. laptops.

---

## Reproduce from scratch

```bash
cd ~/cube-arena && source .venv/bin/activate
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device mps --port 8010 &
python server.py            # then open http://127.0.0.1:8077
```
