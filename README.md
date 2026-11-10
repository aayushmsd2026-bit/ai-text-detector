# AI Text Detector

An open-source detector that estimates how likely a document is to be AI-written. It is a free
website on **GitHub Pages**, and the detection runs **entirely in your browser**, so your text never
leaves your device. A monthly GitHub job retrains it on fresh output from current Claude models, so it
keeps up as the models change.

> **Limits:** this is **not** Turnitin. Turnitin's model is proprietary, and no tool can tell you what
> Turnitin will report. Every AI detector, Turnitin included, sometimes flags human writing
> (especially formal, non-native or heavily edited writing). Treat the score as a signal, never as proof.

## How it works

| Layer | What it measures | Source |
|---|---|---|
| **Binoculars score** | How predictable the text is to one language model, relative to how predictable a second model *expected* it to be. AI text is unusually "expected". | [Hans et al., 2024](https://arxiv.org/abs/2401.12070) |
| **Burstiness** | How much that predictability swings from word to word. Humans vary more. | GLTR / DetectGPT line of work |
| **Style features** | Sentence-length variation, vocabulary variety, stock phrases ("delve", "it's important to note"), transitions, contractions, bullet use, repetition. | `site/js/stylometry.js` |
| **Classifier** | A logistic regression that learns how to weigh all of the above from labelled human and AI documents. | `scripts/train.py` → `site/model.json` |

The two language models are small open models (`SmolLM2-135M` and its instruction-tuned sibling).
GitHub Actions converts them into a single ONNX file (about 270 MB) that the page runs with
[onnxruntime-web](https://onnxruntime.ai/). Visitors download it once, and their browser caches it after that.

The page shows an overall percentage, a sentence-by-sentence highlight and the raw signals. It accepts
pasted text or `.txt`, `.md`, `.docx` and `.pdf` files.

## How it stays up to date

`.github/workflows/retrain.yml` runs on the 1st of every month (or on demand from the **Actions** tab):

1. On the first run, downloads about 200 human-written Wikipedia passages from 2016 (so they can't be AI-written).
2. Asks Claude to write about 30 new documents of mixed types (essays, reports, emails, cover letters, …).
3. Scores only the new files, retrains, and **keeps the old model if the new one is worse**.
4. Commits the data, `site/model.json` and `reports/latest.md` (accuracy, false-positive rate, accuracy by source).
5. Triggers `deploy-pages`, which republishes the site.

## Setup (about 10 minutes, one time)

1. **Turn on GitHub Pages:** in the repo, open **Settings → Pages**. Under *Build and deployment → Source*, choose **GitHub Actions**.
2. **Add your Claude API key:** open **Settings → Secrets and variables → Actions → New repository secret**.
   Name it `ANTHROPIC_API_KEY` and use a key from console.anthropic.com. It's used only for the monthly AI samples.
3. **Publish the site:** open **Actions → deploy-pages → Run workflow**. The first run takes about 10 minutes
   while it converts the models. The site is then live at
   **https://aayushmsd2026-bit.github.io/ai-text-detector/**
4. **Train the first model:** open **Actions → retrain → Run workflow** (about 30–60 minutes the first time). Until it
   finishes, the site works but shows *Uncalibrated*. When it finishes, the site redeploys automatically.
5. **Add your own writing (the biggest accuracy boost):** put documents **you** wrote before using AI into
   `data/human/` as `.txt` files (one per file, 150+ words each). Wikipedia text teaches the detector little
   about how *you* write; 30–100 of your own documents cuts false alarms a lot. Commit them and run
   **retrain** again.

Optional repository **variables** (same settings page, *Variables* tab):

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_MODELS` | `claude-opus-5-5` | Comma-separated Claude models to sample each month |
| `OBSERVER_MODEL` / `PERFORMER_MODEL` | `HuggingFaceTB/SmolLM2-135M` / `-Instruct` | A larger pair, e.g. `SmolLM2-360M` + `SmolLM2-360M-Instruct`, is more accurate, but visitors download about 740 MB on their first visit. Both models must share a tokenizer. |

**Cost:** GitHub Pages and Actions are free for public repos. The monthly Claude sampling is roughly
$1–2 a month on `claude-opus-5-5`.

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
python scripts/export_models.py              # creates site/models/ (one time)
python -m http.server -d site 8000           # open http://localhost:8000
python scripts/train.py                      # retrain from data/
pytest -q                                    # tests
```

## Project layout

```
site/                       the website (static files, published to GitHub Pages)
  index.html, style.css
  js/app.js                 page logic, model download + cache, file reading
  js/detector.js            scoring + classifier (mirrors detector/*.py)
  js/stylometry.js          writing-style features (mirrors detector/stylometry.py)
  model.json                trained classifier (written by retrain)
  models/                   converted language models (built by Actions, not in git)
detector/                   the same detector in Python, used for training
scripts/export_models.py    converts the two language models into one browser ONNX file
scripts/collect_ai_samples.py     fresh samples from the Claude API
scripts/collect_human_samples.py  pre-2022 human text (WikiText-103)
scripts/train.py            trains + evaluates the classifier
data/human, data/ai         training documents
reports/latest.md           latest accuracy report
.github/workflows/          tests, monthly retrain, deploy to GitHub Pages
```

`tests/test_js_parity.py` checks that the browser code and the Python training code give identical
numbers, so the classifier trained in Python behaves the same on the website.

## Measuring it honestly

`reports/latest.md` is measured on held-out documents of the same kinds the model trained on, so real-world
accuracy will be lower. For a tougher, independent test, evaluate against the public
[RAID benchmark](https://github.com/liamdugan/raid), which includes paraphrased and adversarial AI text.
