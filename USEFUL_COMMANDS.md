# Useful Commands

Practical command reference for working with Verify in this repository.

This file complements:
- `README.md` for setup and high-level usage
- `analysis/verify_report.md` for architecture and implementation details

---

## 1. Installation

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url> lantern
cd lantern
```

If submodules were not cloned initially:

```bash
git submodule update --init --recursive
```

Create the main Verify host environment:

```bash
conda create -n lantern python=3.12
conda activate lantern
pip install -r verify/requirements.txt
```

Optional bulk install for native target-app dependencies:

```bash
./install_all_reqs.sh
```

---

## 2. Environment Setup

Create `.env` at repo root with at least:

```env
OPENROUTER_API_KEY=sk-or-...
USE_APP_SERVERS=false
DEBUG=false
VERBOSE=false
```

Typical modes:

```bash
# Serverless mode: fastest path, uses OpenRouter only
export USE_APP_SERVERS=false

# Native mode: runs real app pipelines in isolated conda envs
export USE_APP_SERVERS=true
```

---

## 3. Launch the Streamlit App

```bash
conda activate lantern
streamlit run verify/frontend/app.py
```

---

## 4. Initialization

Use the Streamlit `Initialization` page for per-app env setup, or initialize environments lazily by running the pipeline in native mode.

If you only want serverless mode:

```bash
export USE_APP_SERVERS=false
```

If you want native mode:

```bash
export USE_APP_SERVERS=true
```

---

## 5. Batch Pipeline

Run all enabled rows from `verify/batch_config.csv`:

```bash
python verify/run_batch.py
```

IOC only:

```bash
python verify/run_batch.py --mode ioc
```

Perturbation only:

```bash
python verify/run_batch.py --mode perturb
```

Custom config file:

```bash
python verify/run_batch.py --config verify/batch_config.csv --mode both
```

Parallelize across config rows:

```bash
python verify/run_batch.py --workers 4
```

Limit dataset items per run:

```bash
python verify/run_batch.py --max-items 20
```

Preview without executing:

```bash
python verify/run_batch.py --dry-run
```

Disable cache reuse:

```bash
python verify/run_batch.py --no-cache
```

---

## 6. Re-evaluation

Stamp provenance on existing cached results:

```bash
python verify/reeval.py --init
```

Re-evaluate all output directories with a model:

```bash
python verify/reeval.py --model google/gemini-2.5-pro
```

Re-evaluate only selected apps or datasets:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --app deeptutor xend
python verify/reeval.py --model google/gemini-2.5-pro --dataset PrivacyLens
```

Re-evaluate specific directories:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --dir verify/outputs/cache_d2fbdc5307a7c57b
```

Bare output-directory names are also accepted:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --dir cache_d2fbdc5307a7c57b
```

Dry-run preview:

```bash
python verify/reeval.py --dry-run --model google/gemini-2.5-pro
```

### Prompt variants

Default binary prompt:

```bash
python verify/reeval.py --model google/gemini-2.5-pro
```

MCQ/value-prediction prompt:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt2
```

Channel-wise + aggregate threat prompt:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt3
```

Prompt3 on one cache:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt3 --dir cache_d2fbdc5307a7c57b
```

### Parallel re-eval

Parallelize within a directory:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --workers 4
```

Parallelize across directories too:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --workers 4 --dir-workers 2
```

Two-level parallelism on selected caches:

```bash
python verify/reeval.py \
  --model google/gemini-2.5-pro \
  --prompt3 \
  --workers 4 \
  --dir-workers 2 \
  --dir cache_d2fbdc5307a7c57b cache_44a4eac084755540
```

Note: total OpenRouter concurrency is approximately `workers * dir-workers`.

---

## 7. Evaluation Validation

Populate MCQ predictions for SynthPAI:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt2 --dataset SynthPAI
```

Then open the Streamlit `Eval Validation` page to compare:
- `ext_eval[attr].inferable` vs. ground truth
- `ext_eval[attr].prediction` vs. SynthPAI profile values

---

## 8. IOC / Stage-wise Analysis

Open the Streamlit app and use:
- `Input-Output Comparison` to run IOC live
- `View Input-Output Comparison Results` to inspect cached IOC results

Useful command before opening those pages:

```bash
streamlit run verify/frontend/app.py
```

If you want prompt3-style channel-wise externalization scoring on an IOC cache:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt3 --dir cache_d2fbdc5307a7c57b
```

---

## 9. Cache and Output Inspection

List output directories:

```bash
ls verify/outputs
```

Inspect a run config:

```bash
cat verify/outputs/<run_or_cache_dir>/run_config.json
```

Inspect output-directory summary metadata:

```bash
cat verify/outputs/<run_or_cache_dir>/dir_summary.json
```

Inspect one cached item:

```bash
cat verify/outputs/<run_or_cache_dir>/row_00000.json
```

---

## 10. Sanity Checks

Syntax-check key scripts:

```bash
python -m py_compile verify/reeval.py
python -m py_compile verify/run_batch.py
python -m py_compile verify/frontend/pages/6_Reeval.py
```

Check that `conda` is available:

```bash
conda --version
```

Check that Streamlit can import:

```bash
python -c "import streamlit; print(streamlit.__version__)"
```

---

## 11. Recommended Workflows

Fast serverless workflow:

```bash
conda activate lantern
export USE_APP_SERVERS=false
streamlit run verify/frontend/app.py
```

Batch + re-eval workflow:

```bash
conda activate lantern
python verify/run_batch.py --workers 4
python verify/reeval.py --init
python verify/reeval.py --model google/gemini-2.5-pro --prompt3 --workers 4 --dir-workers 2
```

SynthPAI validation workflow:

```bash
conda activate lantern
python verify/run_batch.py --mode ioc --workers 4
python verify/reeval.py --model google/gemini-2.5-pro --prompt2 --dataset SynthPAI
streamlit run verify/frontend/app.py
```
