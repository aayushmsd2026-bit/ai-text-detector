// Page wiring: loads the model (cached in the browser), reads files, renders results.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { Detector } from "./detector.js";

const REPO_URL = "https://github.com/aayushmsd2026-bit/ai-text-detector";
const CACHE_NAME = "ai-detector-models";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1; // GitHub Pages can't enable the headers multi-threading needs
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("./", location.href).href;

const $ = (id) => document.getElementById(id);
const el = {
  text: $("text"), file: $("file"), count: $("count"), btn: $("analyze"), progress: $("progress"),
  barFill: $("bar-fill"), progressText: $("progress-text"), error: $("error"), empty: $("empty"),
  result: $("result"), pct: $("pct"), label: $("label"), calib: $("calib"), meterFill: $("meter-fill"),
  warnings: $("warnings"), sentences: $("sentences"), signals: $("signals"), json: $("json"),
  download: $("download"), modelInfo: $("model-info"),
};
$("repo-link").href = REPO_URL;

let detectorPromise = null;
let lastReport = null;

const getJSON = async (url) => { try { const r = await fetch(url, { cache: "no-cache" }); return r.ok ? r.json() : null; } catch { return null; } };

function showProgress(fraction, text) {
  el.progress.hidden = false;
  el.barFill.style.width = `${Math.round(fraction * 100)}%`;
  el.progressText.textContent = text;
}

async function fetchModel(url, version) {
  const key = `${url}?v=${version}`;
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* private mode: no cache */ }
  if (cache) {
    const hit = await cache.match(key);
    if (hit) { showProgress(1, "Loading model from browser cache…"); return new Uint8Array(await hit.arrayBuffer()); }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed (${res.status}). Has the site finished deploying?`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    const mb = (n) => (n / 1e6).toFixed(0);
    showProgress(total ? got / total : 0, `Downloading detection model: ${mb(got)}${total ? ` / ${mb(total)}` : ""} MB (one time only)`);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const p of parts) { bytes.set(p, off); off += p.length; }
  if (cache) {
    try {
      for (const req of await cache.keys()) if (req.url.includes(url) && !req.url.endsWith(`v=${version}`)) await cache.delete(req);
      await cache.put(key, new Response(bytes));
    } catch { /* storage full: fine, just re-download next time */ }
  }
  return bytes;
}

async function loadDetector() {
  const [meta, classifier] = await Promise.all([getJSON("models/meta.json"), getJSON("model.json")]);
  if (!meta) throw new Error("Model files not found. Run the deploy-pages workflow on GitHub first.");
  const bytes = await fetchModel("models/binoculars.onnx", meta.sha256 || meta.size_mb);
  showProgress(1, "Starting the model…");
  const [session, tokenizer] = await Promise.all([
    ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }),
    AutoTokenizer.from_pretrained("models"),
  ]);
  return new Detector({ ort, session, tokenizer, classifier });
}

async function readFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".docx")) {
    const { default: mammoth } = await import("https://cdn.jsdelivr.net/npm/mammoth@1.12.3/+esm");
    return (await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() })).value;
  }
  if (name.endsWith(".pdf")) {
    const pdfjs = await import("https://cdn.jsdelivr.net/npm/pdfjs-dist@6.4.299/build/pdf.min.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.4.299/build/pdf.worker.min.mjs";
    const pdf = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
    const pages = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const content = await (await pdf.getPage(i)).getTextContent();
      pages.push(content.items.map((it) => it.str + (it.hasEOL ? "\n" : "")).join(""));
    }
    return pages.join("\n\n");
  }
  return file.text();
}

const SIGNALS = [
  ["binoculars", "Binoculars score (lower = more AI-like)", 3],
  ["log_ppl", "Perplexity (log): how surprising the words are", 2],
  ["log_ppl_std", "Burstiness: how much surprise varies", 2],
  ["sent_len_cv", "Sentence-length variation", 2],
  ["stock_phrase_rate", "Stock AI phrases per 100 words", 2],
  ["type_token_ratio", "Vocabulary variety", 2],
  ["contraction_rate", "Contractions per 100 words", 2],
  ["n_words", "Words analysed", 0],
];

function band(p) {
  if (p === null) return "none";
  return p >= 0.7 ? "ai" : p >= 0.4 ? "mixed" : "human";
}

function render(r) {
  const pct = Math.round(r.ai_probability * 100);
  el.empty.hidden = true;
  el.result.hidden = false;
  el.pct.textContent = `${pct}%`;
  el.label.textContent = `AI likelihood: ${r.label}`;
  const m = r.meta;
  el.calib.textContent = r.calibrated
    ? `Classifier trained ${m?.trained_at?.slice(0, 10) ?? ""} on ${m?.n_human ?? "?"} human + ${m?.n_ai ?? "?"} AI documents · test AUC ${m?.auc?.toFixed(3) ?? "?"}`
    : "⚠️ Uncalibrated: no trained classifier yet";
  el.meterFill.style.width = `${pct}%`;
  el.meterFill.style.background = `var(--${band(r.ai_probability)}-solid)`;
  el.warnings.replaceChildren(...r.warnings.map((w) => Object.assign(document.createElement("li"), { textContent: w })));

  const frag = document.createDocumentFragment();
  let pos = 0;
  for (const s of r.sentences) {
    if (s.start > pos) frag.append(r.text.slice(pos, s.start));
    const mark = document.createElement("mark");
    mark.className = band(s.ai_probability);
    mark.textContent = s.text;
    mark.title = s.ai_probability === null ? "Too short to judge" : `${Math.round(s.ai_probability * 100)}% AI-like`;
    frag.append(mark);
    pos = s.end;
  }
  if (pos < r.text.length) frag.append(r.text.slice(pos));
  el.sentences.replaceChildren(frag);

  el.signals.replaceChildren(...SIGNALS.map(([k, label, d]) => {
    const tr = document.createElement("tr");
    const v = r.features[k];
    tr.append(Object.assign(document.createElement("td"), { textContent: label }),
      Object.assign(document.createElement("td"), { textContent: Number.isFinite(v) ? v.toFixed(d) : "–" }));
    return tr;
  }));

  lastReport = {
    ai_probability: +r.ai_probability.toFixed(4), label: r.label, calibrated: r.calibrated, warnings: r.warnings,
    features: Object.fromEntries(Object.entries(r.features).map(([k, v]) => [k, Number.isFinite(v) ? +v.toFixed(4) : null])),
    sentences: r.sentences.map((s) => ({ text: s.text, ai_probability: s.ai_probability === null ? null : +s.ai_probability.toFixed(3) })),
    analysed_at: new Date().toISOString(),
  };
  el.json.textContent = JSON.stringify(lastReport, null, 2);
}

function updateCount() {
  const n = (el.text.value.match(/[A-Za-z][A-Za-z'-]*/g) || []).length;
  el.count.textContent = `${n} words`;
}

el.text.addEventListener("input", updateCount);

el.file.addEventListener("change", async () => {
  const f = el.file.files[0];
  if (!f) return;
  el.error.hidden = true;
  try { el.text.value = await readFile(f); updateCount(); }
  catch (e) { el.error.textContent = `Couldn't read that file: ${e.message}`; el.error.hidden = false; }
});

el.btn.addEventListener("click", async () => {
  const text = el.text.value.trim();
  el.error.hidden = true;
  if (!text) { el.error.textContent = "Paste some text or upload a file first."; el.error.hidden = false; return; }
  el.btn.disabled = true;
  try {
    detectorPromise ??= loadDetector();
    const det = await detectorPromise;
    const r = await det.analyze(text, (done, total) => showProgress(total ? done / total : 1, `Analysing… part ${Math.min(done + 1, total)} of ${total}`));
    r.meta = det.clf?.meta;
    render(r);
    el.progress.hidden = true;
  } catch (e) {
    detectorPromise = null;
    el.error.textContent = e.message || String(e);
    el.error.hidden = false;
    el.progress.hidden = true;
  } finally {
    el.btn.disabled = false;
  }
});

el.download.addEventListener("click", () => {
  if (!lastReport) return;
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" })),
    download: "ai-detector-report.json",
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

(async () => {
  const [meta, clf] = await Promise.all([getJSON("models/meta.json"), getJSON("model.json")]);
  const parts = [];
  if (meta) parts.push(`Models: ${meta.observer} + ${meta.performer} (${meta.size_mb} MB download).`);
  parts.push(clf ? `Classifier last retrained ${clf.meta.trained_at.slice(0, 10)}.` : "Classifier not trained yet.");
  el.modelInfo.textContent = parts.join(" ");
})();
