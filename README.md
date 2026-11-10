---
title: AI Text Detector
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# AI Text Detector

An open-source detector that estimates how likely a document is to be AI-written.
It retrains itself every month on fresh output from current Claude models, so it keeps up as the
models change.

> **Limits:** this is **not** Turnitin. Turnitin's model is proprietary, and no tool can tell you what
> Turnitin will report. Every AI detector, Turnitin included, sometimes flags human writing
> (especially formal, non-native or heavily edited writing). Treat the score as a signal, never as proof.

## How it works

| Layer | What it measures | Source |
|---|---|---|
| **Binoculars score** | How predictable the text is to one language model, relative to how predictable a second model *expected* it to be. AI text is unusually "expected". | [Hans et al., 2024](https://arxiv.org/abs/2401.12070) |
| **Burstiness** | How much that predictability swings from token to token. Humans vary more. | GLTR / DetectGPT line of work |
| **Style features** | Sentence-length variation, vocabulary richness, stock phrases ("delve", "it's important to note"), transitions, contractions, bullet use, repetition. | `detector/stylometry.py` |
| **Classifier** | A calibrated model that learns how to weigh all of the above from labelled human and AI documents. | `scripts/train.py` |

The language models are small and open (`Qwen/Qwen2.5-0.5B` + `Qwen/Qwen2.5-0.5B-Instruct`), so the app
runs on a free CPU host. With a GPU you can set `OBSERVER_MODEL` / `PERFORMER_MODEL` to a larger pair that
shares a tokenizer (for example `Qwen/Qwen2.5-7B` + `Qwen/Qwen2.5-7B-Instruct`) for better accuracy.

The web page shows an overall percentage, a sentence-by-sentence highlight and the raw signals. It accepts
pasted text or `.txt`, `.md`, `.docx` and `.pdf` files.

## How it stays up to date

`.github/workflows/retrain.yml` runs on the 1st of every month (or on demand from the **Actions** tab):

1. On the first run, downloads ~200 human-written Wikipedia passages from 2016 (so they can't be AI-written).
2. Asks Claude to write ~30 new documents of mixed types (essays, reports, emails, cover letters, …).
3. Scores only the new files, retrains, and **keeps the old model if the new one is worse**.
4. Commits the data, model and `reports/latest.md` (accuracy, false-positive rate, accuracy by source).
5. Redeploys the website.

## Setup (about 15 minutes, one time)

### 1. Create the Hugging Face Space (free hosting)

1. Make a free account at huggingface.co.
2. **New Space** → name it (e.g. `ai-text-detector`) → SDK **Gradio** → hardware **CPU basic (free)** → Create.
3. **Settings → Access Tokens** → create a token with **write** access. Copy it.

### 2. Connect GitHub to the Space

In this GitHub repo: **Settings → Secrets and variables → Actions**.

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | the Hugging Face write token |
| Variable | `HF_SPACE` | `your-hf-username/ai-text-detector` |
| Secret | `ANTHROPIC_API_KEY` | an API key from console.anthropic.com (for the monthly AI samples) |
| Variable *(optional)* | `CLAUDE_MODELS` | comma-separated models to sample, default `claude-opus-5-5` |

Then **Actions → deploy-space → Run workflow**. The site appears at
`https://huggingface.co/spaces/<you>/ai-text-detector` after the Space finishes building (~5 min).

### 3. Train the first model

**Actions → retrain → Run workflow.** The first run takes longest (it downloads models and scores ~230
documents on CPU). Until it finishes, the site works but shows *Uncalibrated*.

### 4. Add your own writing (biggest accuracy boost)

Put documents **you** wrote before you used AI into `data/human/` as `.txt` files (one per file, 150+
words). Generic Wikipedia text teaches the detector little about how *you* write. 30–100 of your own
documents makes a big difference to false positives. Commit them and run **retrain** again.

Cost: the monthly Claude sampling is about 30 short documents per model, roughly $1 to $2 a month on
`claude-opus-5-5`. Everything else is free.

## Run locally

```bash
pip install -r requirements-train.txt
python app.py                       # web UI at http://127.0.0.1:7860
python scripts/train.py             # retrain from data/
pytest -q                           # tests (no model download needed)
```

## Project layout

```
app.py                      web interface (Gradio)
detector/lm_scores.py       Binoculars scoring with two open models
detector/stylometry.py      writing-style features
detector/pipeline.py        combines everything into a verdict
scripts/collect_ai_samples.py     fresh samples from the Claude API
scripts/collect_human_samples.py  pre-2022 human text (WikiText-103)
scripts/train.py            trains + evaluates the classifier
data/human, data/ai         training documents
model/classifier.joblib     the trained model (created by retrain)
reports/latest.md           latest accuracy report
.github/workflows/          tests, monthly retrain, deploy to Hugging Face
```

## Measuring it honestly

`reports/latest.md` is measured on held-out documents of the same kinds the model trained on, so real-world
accuracy will be lower. For a tougher, independent test, evaluate against the public
[RAID benchmark](https://github.com/liamdugan/raid), which includes paraphrased and adversarial AI text.
