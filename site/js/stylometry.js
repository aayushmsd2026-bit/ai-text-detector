// Writing-style features. Line-for-line port of detector/stylometry.py:
// tests/test_js_parity.py checks both give identical numbers.

const SENT_SPLIT = /(?<=[.!?])\s+(?=["'(\[]?[A-Z0-9])/;
const WORD = /[A-Za-z][A-Za-z'-]*/g;
const PARA = /[^\n]+(?:\n(?!\s*\n)[^\n]+)*/g;
const LINE_BREAK = /\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]/;

export const STOCK_PHRASES = [
  "delve", "tapestry", "testament to", "in today's", "it's important to note",
  "it is important to note", "plays a crucial role", "plays a vital role",
  "navigate the", "landscape of", "multifaceted", "furthermore", "moreover",
  "in conclusion", "overall,", "additionally,", "seamless", "robust",
  "leverage", "foster", "holistic", "pivotal", "underscore", "showcase",
  "a myriad of", "ever-evolving", "not only", "nuanced", "realm",
];
const TRANSITIONS = ["however", "therefore", "furthermore", "moreover", "additionally",
  "consequently", "nevertheless", "thus", "hence", "ultimately"];

export const FEATURE_NAMES = [
  "n_words", "sent_len_mean", "sent_len_std", "sent_len_cv",
  "type_token_ratio", "hapax_ratio", "avg_word_len",
  "stock_phrase_rate", "transition_rate", "comma_rate",
  "em_dash_rate", "colon_rate", "bullet_line_ratio",
  "repeated_trigram_rate", "first_person_rate", "contraction_rate",
];

const words = (s) => s.match(WORD) || [];
const countOf = (hay, needle) => hay.split(needle).length - 1;
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) ** 2))); };

/** Returns [{start, end, text}] sentence spans, also splitting on blank lines. */
export function splitSentences(text) {
  const spans = [];
  for (const para of text.matchAll(PARA)) {
    const pStart = para.index, pText = para[0];
    let pos = 0;
    for (const piece of pText.split(SENT_SPLIT)) {
      const idx = pText.indexOf(piece, pos);
      if (piece.trim()) spans.push({ start: pStart + idx, end: pStart + idx + piece.length, text: piece });
      pos = idx + piece.length;
    }
  }
  return spans;
}

export function features(text) {
  const ws = words(text).map((w) => w.toLowerCase());
  const n = Math.max(ws.length, 1);
  const sentLens = splitSentences(text).map((s) => words(s.text).length).filter((l) => l > 0);
  const lens = sentLens.length ? sentLens : [0];
  const counts = new Map();
  for (const w of ws) counts.set(w, (counts.get(w) || 0) + 1);
  const c = (w) => counts.get(w) || 0;
  const lower = text.toLowerCase();
  const tri = new Map();
  for (let i = 0; i + 2 < ws.length; i++) {
    const k = ws[i] + " " + ws[i + 1] + " " + ws[i + 2];
    tri.set(k, (tri.get(k) || 0) + 1);
  }
  const nTri = Math.max(ws.length - 2, 0);
  const lines = text.split(LINE_BREAK).filter((l) => l.trim());
  const per100 = (x) => (100 * x) / n;
  const m = mean(lens);

  let hapax = 0;
  for (const v of counts.values()) if (v === 1) hapax++;
  let repeated = 0;
  for (const v of tri.values()) if (v > 1) repeated += v;

  return {
    n_words: ws.length,
    sent_len_mean: m,
    sent_len_std: std(lens),
    sent_len_cv: m ? std(lens) / m : 0,
    type_token_ratio: counts.size / n,
    hapax_ratio: hapax / n,
    avg_word_len: ws.length ? mean(ws.map((w) => w.length)) : 0,
    stock_phrase_rate: per100(STOCK_PHRASES.reduce((s, p) => s + countOf(lower, p), 0)),
    transition_rate: per100(TRANSITIONS.reduce((s, t) => s + c(t), 0)),
    comma_rate: per100(countOf(text, ",")),
    em_dash_rate: per100(countOf(text, "—") + countOf(text, " -- ")),
    colon_rate: per100(countOf(text, ":")),
    bullet_line_ratio: lines.length ? lines.filter((l) => /^\s*([-*•]|\d+[.)])\s/.test(l)).length / lines.length : 0,
    repeated_trigram_rate: nTri ? repeated / nTri : 0,
    first_person_rate: per100(["i", "me", "my", "mine", "i'm", "i've", "i'd"].reduce((s, w) => s + c(w), 0)),
    contraction_rate: per100(ws.filter((w) => w.includes("'")).length),
  };
}
