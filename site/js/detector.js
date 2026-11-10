// Browser detector. Mirrors detector/lm_scores.py + detector/pipeline.py.
// Libraries (onnxruntime-web, Transformers.js tokenizer) are passed in so the
// same code runs in the page and in the Node parity test.

import { features as styleFeatures, splitSentences } from "./stylometry.js";

export const CHUNK = 512; // must match CHUNK in detector/lm_scores.py
export const MIN_WORDS = 60;
const FALLBACK_THRESHOLD = 0.95;
const FALLBACK_SCALE = 0.05;

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(Math.min(x, 50), -50)));
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

export function labelFor(p) {
  if (p >= 0.8) return "Likely AI-generated";
  if (p >= 0.6) return "Possibly AI-generated / heavily AI-assisted";
  if (p > 0.4) return "Uncertain";
  if (p > 0.2) return "Probably human-written";
  return "Likely human-written";
}

/** Character span of each token, found by decoding tokens in order.
 *  Leading whitespace is excluded so a token belongs to the sentence it starts. */
export function tokenOffsets(ids, decode) {
  const offsets = [];
  let pos = 0, group = [];
  for (const id of ids) {
    group.push(id);
    const piece = decode(group);
    if (piece.endsWith("�") && group.length < 4) continue; // partial UTF-8 character
    const lead = piece.length - piece.trimStart().length;
    for (let k = 0; k < group.length; k++) offsets.push([pos + lead, pos + piece.length]);
    pos += piece.length;
    group = [];
  }
  for (let k = 0; k < group.length; k++) offsets.push([pos, pos]);
  return offsets;
}

export function summary(lp, xp) {
  if (!lp.length) return { binoculars: NaN, log_ppl: NaN, x_ppl: NaN, log_ppl_std: NaN };
  const l = mean(lp), x = mean(xp);
  return {
    binoculars: x > 0 ? l / x : NaN,
    log_ppl: l,
    x_ppl: x,
    log_ppl_std: Math.sqrt(mean(lp.map((v) => (v - l) ** 2))),
  };
}

export function spanScore(scores, start, end) {
  let l = 0, x = 0, n = 0;
  scores.offsets.forEach(([s, e], i) => {
    if (s >= start && e <= end && e > s) { l += scores.lp[i]; x += scores.xp[i]; n++; }
  });
  return n < 3 || x <= 0 ? null : l / x;
}

export function classifierProb(clf, feats) {
  let z = clf.intercept;
  clf.features.forEach((name, i) => {
    let v = feats[name];
    if (v === undefined || v === null || Number.isNaN(v)) v = 0;
    z += clf.coef[i] * (v - clf.mean[i]) / (clf.scale[i] || 1);
  });
  return sigmoid(z);
}

export class Detector {
  /** @param deps {ort, session, tokenizer, classifier|null} */
  constructor({ ort, session, tokenizer, classifier }) {
    this.ort = ort;
    this.session = session;
    this.tokenizer = tokenizer;
    this.clf = classifier || null;
  }

  decode(ids) {
    return this.tokenizer.decode(ids, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
  }

  async score(text, onProgress = () => {}) {
    const ids = this.tokenizer.encode(text, { add_special_tokens: false });
    const offsets = tokenOffsets(ids, (g) => this.decode(g));
    const lp = [], xp = [], kept = [];
    const nChunks = Math.ceil(ids.length / CHUNK);
    for (let start = 0, c = 0; start < ids.length; start += CHUNK, c++) {
      onProgress(c, nChunks);
      const chunk = ids.slice(start, start + CHUNK);
      if (chunk.length < 2) continue;
      const input = new this.ort.Tensor("int64", BigInt64Array.from(chunk.map(BigInt)), [1, chunk.length]);
      const out = await this.session.run({ input_ids: input });
      lp.push(...out.log_ppl.data);
      xp.push(...out.x_ppl.data);
      kept.push(...offsets.slice(start + 1, start + chunk.length));
      await new Promise((r) => setTimeout(r, 0)); // let the page repaint
    }
    onProgress(nChunks, nChunks);
    return { lp, xp, offsets: kept };
  }

  sentenceProb(bino) {
    const thr = this.clf?.sentence_threshold ?? FALLBACK_THRESHOLD;
    const scale = this.clf?.sentence_scale ?? FALLBACK_SCALE;
    return sigmoid((thr - bino) / scale);
  }

  async analyze(rawText, onProgress) {
    const text = rawText.trim();
    const warnings = [];
    const scores = await this.score(text, onProgress);
    const feats = { ...summary(scores.lp, scores.xp), ...styleFeatures(text) };
    if (feats.n_words < MIN_WORDS) {
      warnings.push(`Only ${feats.n_words} words. Results under ${MIN_WORDS} words are unreliable.`);
    }
    let p;
    if (this.clf) {
      p = classifierProb(this.clf, feats);
    } else {
      warnings.push("No trained classifier yet: using the uncalibrated Binoculars threshold only.");
      p = Number.isNaN(feats.binoculars) ? 0.5 : this.sentenceProb(feats.binoculars);
    }
    const sentences = splitSentences(text).map((s) => {
      const b = spanScore(scores, s.start, s.end);
      return { ...s, ai_probability: b === null ? null : this.sentenceProb(b) };
    });
    return { text, ai_probability: p, label: labelFor(p), calibrated: !!this.clf, features: feats, sentences, warnings };
  }
}
