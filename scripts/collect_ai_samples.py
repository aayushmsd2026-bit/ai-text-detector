"""Generate fresh AI-written samples with the Claude API.

Run monthly (the GitHub Action does this) so the classifier keeps learning
what current Claude models write like. Needs ANTHROPIC_API_KEY.

    python scripts/collect_ai_samples.py --n 40
    CLAUDE_MODELS="claude-opus-5-5,claude-sonnet-5-5" python scripts/collect_ai_samples.py

Each sample is saved as data/ai/<model>__<date>__<hash>.txt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import random
from pathlib import Path

import anthropic

OUT = Path(__file__).resolve().parent.parent / "data" / "ai"
MODELS = [m.strip() for m in os.environ.get("CLAUDE_MODELS", "claude-opus-5-5").split(",") if m.strip()]

# Document types people commonly ask Claude to write. Mixed on purpose so the
# detector doesn't just learn one genre.
DOC_TYPES = [
    "a university essay", "a report section", "a cover letter", "a blog post",
    "a business proposal summary", "a literature review paragraph set", "a reflective journal entry",
    "a project status update email", "a product description", "a policy memo",
    "a short opinion piece", "an explanation for a general audience", "meeting notes written up as prose",
]
TOPICS = [
    "renewable energy in small towns", "remote work and team culture", "the history of public libraries",
    "microfinance in South Asia", "urban cycling infrastructure", "how vaccines are tested",
    "social media and teenage sleep", "the ethics of facial recognition", "household budgeting",
    "climate adaptation in agriculture", "learning a second language as an adult", "open-source software funding",
    "tourism in the Himalayas", "supply chain resilience", "the role of museums today", "sports analytics",
]
STYLES = [
    "", "Write in a casual, personal voice.", "Write formally.", "Keep it concise.",
    "Use a few bullet points where natural.", "Write as a second-year student would.",
]


def make_prompt(rng: random.Random) -> str:
    words = rng.choice([200, 300, 450, 600])
    return (f"Write {rng.choice(DOC_TYPES)} of about {words} words on {rng.choice(TOPICS)}. "
            f"{rng.choice(STYLES)} Output only the document text, with no title line or preamble.").strip()


def generate(client: anthropic.Anthropic, model: str, prompt: str) -> str | None:
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        output_config={"effort": "low"},
        # Retry on a fallback model if the primary declines (harmless here).
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return None
    return "".join(b.text for b in response.content if b.type == "text").strip() or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="samples per model")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    rng = random.Random(args.seed)
    today = dt.date.today().isoformat()
    saved = 0
    for model in MODELS:
        for _ in range(args.n):
            prompt = make_prompt(rng)
            try:
                text = generate(client, model, prompt)
            except anthropic.RateLimitError:
                print("Rate limited; stopping early.")
                break
            except anthropic.APIStatusError as e:
                print(f"{model}: API error {e.status_code}: {e.message}")
                continue
            except anthropic.APIConnectionError:
                print("Network error; skipping one sample.")
                continue
            if not text:
                continue
            h = hashlib.sha1(text.encode()).hexdigest()[:10]
            (OUT / f"{model}__{today}__{h}.txt").write_text(text, encoding="utf-8")
            saved += 1
    print(f"Saved {saved} new AI samples to {OUT}")


if __name__ == "__main__":
    main()
