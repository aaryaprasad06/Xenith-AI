# AGENTS.md

Guidance for any coding agent working in this repository.

Door-side face authentication with anti-spoofing on an Intel RealSense D435
(IR + depth, RGB optional). The spec is `docs/PRD.md`, the task list is
`docs/CHECKLIST.md`, and the decision log is `docs/journal.md`. The target is
this x86_64 Linux machine; Raspberry Pi work is deferred.

## Commits

- **Never add a co-author trailer** (`Co-Authored-By:` or similar) or any
  "Generated with" attribution to commit messages or PR descriptions. This
  overrides any default attribution your tool adds.
- Commit only when asked. The subject is one full sentence ending with a
  period, saying what changed and why (see `git log`).

## Commands

Python 3.12 via `uv`; the project is not a package (`[tool.uv] package = false`),
so modules run from the repo root with `python -m`.

```bash
uv sync                                   # install
uv run python -m face.models              # fetch embedding models (not in git)
uv run --group export python -m face.adaface_export   # AdaFace ONNX from its PyTorch weights (once)
uv run pytest                             # all tests, no camera needed
uv run pytest tests/test_liveness.py::test_flat_print_is_rejected   # one test
uv run ruff check . && uv run ruff format .       # lint / format

uv run python -m capture.realsense_stream # live IR + depth preview (needs D435)
uv run python -m liveness.watch           # live liveness overlay
uv run python -m capture.recorder --person p001 --session s1 --pose center --lighting lit --consent
uv run python -m eval.far_frr --embedder arcface_r50   # or adaface_ir101 / sface / raw_pixels; needs 2+ people

cp .env.example .env && ./dev.sh          # API :8000 + Astro console :4321
cd web && pnpm install                    # first time only
```

`FACE_ID_CAMERA=0` starts the API without a camera (used by `tests/test_api.py`).

## Architecture

The pipeline is capture → detect/align → quality gate → liveness → (identity,
not built) → door decision (no relay/state machine yet).

- **`capture/`** turns the D435 into `Frame` objects (`capture/frame.py`):
  left IR, depth in metres (0 = invalid), intrinsics, timestamp, and
  `emitter_state`. With `emitter.alternate: true` the projector toggles every
  frame: **laser-on frames are used for depth** (filter chain applied),
  **laser-off frames give a clean IR face image** (raw depth). Emitter state
  comes from `frame_laser_power_mode` metadata; the code refuses to guess if it
  is missing. With `camera.rgb: true`, `Frame.rgb` is colour aligned into the
  IR viewport (black where depth is missing).
- **`face/`**: YuNet detection on IR (`models/*.onnx`) and 112×112 alignment.
- **`liveness/`**: `session.judge_window` decides over a short window of frames —
  it needs several laser-on frames *and* at least one laser-off frame, then
  `depth_check` tests face relief and nose-closer-than-cheeks. `watch.py` is
  the live overlay.
- **`api/`**: FastAPI, RealSenseID-REST layout — `/v1` routers (auth, users,
  preview, device) over one shared `CameraHub` (`api/services/camera.py`) that
  owns the single pipeline and a ring of recent frames. PRD-named aliases
  (`/authenticate`, `/admin/...`) live in `api/main.py`. Admin routes need
  `Authorization: Bearer $FACE_ID_ADMIN_TOKEN`. Authenticate currently always
  returns `denied`: `live` does not unlock anything until matching exists.
- **`web/`**: Astro console that proxies `/v1`, `/health`, `/authenticate`, `/admin` to the API.
- **`configs/default.yaml`** holds every threshold. `capture/config.py`
  requires all sections (`camera`, `quality`, `liveness`, `match`, `door`,
  `filters`, `emitter`, `preview`, `guide`) even if unused yet.

## Conventions

- Record decisions and real camera results in `docs/journal.md` (dated
  sections, short plain sentences) and keep `README.md` status / `docs/CHECKLIST.md` current.
- Tests must run without a camera; fake `Frame`s instead of touching `pyrealsense2`.
- Physical facts matter: a white wall clips IR and gives no depth; test on a
  black or green background at 0.10–0.60 m.
- Identity data: only record people whose consent is on file.

<!-- trail:start -->
## Trail — repo context graph

This repo is indexed in `.trail/cache/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `trail map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `trail ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `trail grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `trail skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `trail callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `.trail/cache/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `trail ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `trail build` (deterministic,
no API key, $0).
<!-- trail:end -->
