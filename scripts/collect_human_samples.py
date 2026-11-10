"""Download human-written text from before chatbots existed.

Pulls paragraphs from WikiText-103 (Wikipedia, 2016) so they cannot be AI
generated. This is only a starting point: Wikipedia is one style. Your own
pre-2022 essays, reports and emails in data/human/ improve results far more.

    python scripts/collect_human_samples.py --n 200
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "human"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    # Re-join lines into articles' body sections (headings look like " = Title = ").
    docs, cur = [], []
    for line in ds["text"]:
        if re.match(r"^\s*=.*=\s*$", line):
            if cur:
                docs.append(" ".join(cur))
                cur = []
        elif line.strip():
            cur.append(line.strip())
    rng = random.Random(args.seed)
    rng.shuffle(docs)

    OUT.mkdir(parents=True, exist_ok=True)
    saved = 0
    for d in docs:
        # Undo WikiText's tokenisation artefacts (" @-@ ", " , ") so it reads like normal prose.
        d = d.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        d = re.sub(r" ([.,;:!?)'])", r"\1", d).replace("( ", "(")
        words = d.split()
        if len(words) < args.min_words:
            continue
        d = " ".join(words[:600])
        h = hashlib.sha1(d.encode()).hexdigest()[:10]
        (OUT / f"wikitext__{h}.txt").write_text(d, encoding="utf-8")
        saved += 1
        if saved >= args.n:
            break
    print(f"Saved {saved} human samples to {OUT}")


if __name__ == "__main__":
    main()
