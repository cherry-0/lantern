# AGENTS.md

Guidance for Codex (Codex.ai/code) when working in this repository.

## Overview

Lantern is a privacy evaluation framework (`verify/`) wired up to a collection of
target apps (`target-apps/`). Each target app gets an adapter that lets the
framework feed it inputs, capture outputs, and score privacy leakage.

Repo layout:
- `target-apps/` — one subdirectory per app (27 apps; some are git submodules)
- `verify/backend/adapters/` — one adapter per app, registered in `__init__.py`
- `verify/backend/` — `runners/`, `drivers/`, `evaluation_method/`, `judge/`,
  `observers/`, `perturbation_method/`, `datasets/`, `orchestrator.py`
- `verify/frontend/` — Streamlit app (`app.py` + `pages/`)
- `verify/outputs/` — experiment results, one directory per run
- `analysis/` — writeups for apps (`app_descriptions/<app_name>.md`), `scripts/` for analysis code,
  `attachments/` for figures, `results/` for exported PDFs
- `paper/`, `models/`, `cache/`, `scripts/` — supporting assets; ignore unless asked

## Key references

> **Adding a new app to `verify/`?**
> See [`analysis/verify_report.md`](analysis/verify_report.md) — Section 7
> ("How to Add a New App") has step-by-step templates for the adapter, runner,
> Django settings shim, adapter registration, and UI activation.

> **Hitting a `verify/` error?**
> See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — known runner failures,
> phase-drift bugs, Django config issues, conda env problems, frontend quirks.

> **Running experiments?**
> See [`USEFUL_COMMANDS.md`](USEFUL_COMMANDS.md) — install, env vars, batch
> pipeline, re-evaluation, IOC, cache inspection, adb recipes.

> **Writing the paper?**
> See [`paper/PAPER_WRITING.md`](paper/PAPER_WRITING.md) — build commands,
> writing-style conventions, plotting / figure-layout rules, and useful
> commands like `/write-paper-summary` and `/refine-paper-flow`. Plot style
> details live in [`analysis/FIGURE.md`](analysis/FIGURE.md).

## Workflows

**Implementation.** Setup and run:
1. Activate the env: `conda activate lantern` (Python 3.12, deps in `verify/requirements.txt`).
2. Launch the UI: `streamlit run verify/frontend/app.py`.
3. Adapter changes go in `verify/backend/adapters/<app>.py` and must be registered
   in `verify/backend/adapters/__init__.py`.
4. Document app-level changes in `analysis/app_descriptions/<app_name>.md`. If the overall
   structure of `verify/` has changed, update `analysis/verify_report.md`.

**Analysis.** When asked for an experiment analysis:
1. Put/update Python in `analysis/scripts/` — most building blocks already exist
   under `verify/`, copy and adapt rather than rewriting from scratch.
2. Document findings in `analysis/<topic>.md`.
3. Follow the `/deep-analysis` skill for methodology.
4. Follow [`analysis/FIGURE.md`](analysis/FIGURE.md) for any plots.
   Experiment outputs to read from: `verify/outputs/`.

**Writing.** When asked to write a paper section:
1. Find the corresponding section in `paper/26_CCS_Lantern/sections/`.
2. Read findings in `analysis/<topic>.md` (primary) and exported PDFs in
   `analysis/results/`.
3. Write/update the section in `paper/26_CCS_Lantern/sections/` following the
   conventions in [`paper/PAPER_WRITING.md`](paper/PAPER_WRITING.md) (voice,
   verdict / channel typography, when to inline numbers vs. defer to a table).
4. If needed, update the bibliography: `paper/26_CCS_Lantern/reference.bib`
   (papers) or `software.bib` (tools/software).
5. Compile via `paper/26_CCS_Lantern/build.sh` (uses `latexmk` if available;
   builds into `build/`, copies `main.pdf` to the directory root). Check
   `build/main.log` for errors.
6. Refresh the Korean section summary with `/write-paper-summary` if the
   structure changed.