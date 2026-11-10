"""Web interface (Gradio). Runs locally with `python app.py` or on Hugging Face Spaces."""

from __future__ import annotations

import gradio as gr

from detector.pipeline import Detector, result_to_json
from detector.textio import read_any

detector = Detector()

DISCLAIMER = (
    "**Read this first.** No AI detector is reliable enough to prove anything. This tool "
    "estimates a probability from open methods (Binoculars + writing-style features + a "
    "classifier retrained over time). It is **not** Turnitin and cannot predict what Turnitin "
    "will report. Formal, non-native or heavily edited human writing is often flagged by mistake."
)


def _highlight(text: str, sentences) -> list[tuple[str, str | None]]:
    out, pos = [], 0
    for s in sentences:
        if s.start > pos:
            out.append((text[pos:s.start], None))
        p = s.ai_probability
        tag = None if p is None else ("AI-like" if p >= 0.7 else "Mixed" if p >= 0.4 else "Human-like")
        out.append((s.text, tag))
        pos = s.end
    if pos < len(text):
        out.append((text[pos:], None))
    return out


def run(text: str, file):
    if file is not None:
        text = read_any(file if isinstance(file, str) else file.name)
    text = (text or "").strip()
    if not text:
        return "Paste some text or upload a file.", None, None, None
    r = detector.analyze(text)
    pct = round(100 * r.ai_probability)
    summary = f"## {pct}% AI likelihood: {r.label}\n"
    summary += "Calibrated classifier" if r.calibrated else "⚠️ Uncalibrated (no trained classifier yet)"
    if r.model_info.get("trained_at"):
        summary += f" · trained {r.model_info['trained_at'][:10]}"
        if "auc" in r.model_info:
            summary += f" · held-out AUC {r.model_info['auc']:.3f}"
    for w in r.warnings:
        summary += f"\n\n> {w}"
    key = {k: round(v, 3) for k, v in r.features.items() if k in
           ("binoculars", "log_ppl", "log_ppl_std", "sent_len_cv", "stock_phrase_rate", "type_token_ratio")}
    return summary, _highlight(text, r.sentences), key, result_to_json(r)


with gr.Blocks(title="AI Text Detector") as demo:
    gr.Markdown("# AI Text Detector")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(lines=14, label="Text", placeholder="Paste at least ~150 words for a meaningful result")
            up = gr.File(label="…or upload .txt / .md / .docx / .pdf", file_types=[".txt", ".md", ".docx", ".pdf"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_summary = gr.Markdown()
            out_hl = gr.HighlightedText(label="Sentence view", combine_adjacent=False, show_legend=True,
                                        show_inline_category=False,
                                        color_map={"AI-like": "red", "Mixed": "orange", "Human-like": "green"})
            out_feats = gr.JSON(label="Key signals")
            with gr.Accordion("Full report (JSON)", open=False):
                out_json = gr.Code(language="json")
    btn.click(run, [inp, up], [out_summary, out_hl, out_feats, out_json])

if __name__ == "__main__":
    demo.launch()
