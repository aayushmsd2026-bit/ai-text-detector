"""Convert the two language models into one browser-ready ONNX file.

The output graph takes `input_ids` and returns, for every position after the
first, the two Binoculars ingredients:

    log_ppl[t] = -log P_performer(token[t+1])
    x_ppl[t]   = -sum_v P_observer(v) * log P_performer(v)

Putting both models in one graph means the browser downloads a single file
and never has to move vocabulary-sized logits around, and the training job
(Python, onnxruntime) runs the exact same computation as the website.

    python scripts/export_models.py --out site/models
    OBSERVER_MODEL=... PERFORMER_MODEL=... python scripts/export_models.py

Writes binoculars.onnx (int8-quantised), tokenizer files and meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_OBSERVER = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_PERFORMER = "HuggingFaceTB/SmolLM2-135M-Instruct"


class Binoculars(torch.nn.Module):
    def __init__(self, observer, performer):
        super().__init__()
        self.observer, self.performer = observer, performer

    def forward(self, input_ids):
        # An explicit all-ones mask keeps transformers off its packed-sequence
        # detection path, which uses an operator ONNX can't export (aten::diff).
        mask = torch.ones_like(input_ids)
        obs = self.observer(input_ids=input_ids, attention_mask=mask, use_cache=False).logits[0, :-1].float()
        perf = self.performer(input_ids=input_ids, attention_mask=mask, use_cache=False).logits[0, :-1].float()
        perf_logp = torch.log_softmax(perf, dim=-1)
        targets = input_ids[0, 1:]
        log_ppl = -perf_logp.gather(-1, targets[:, None]).squeeze(-1)
        x_ppl = -(torch.softmax(obs, dim=-1) * perf_logp).sum(-1)
        return log_ppl, x_ppl


TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]


def copy_tokenizer_files(src: str, out: Path) -> None:
    if Path(src).is_dir():
        for name in TOKENIZER_FILES:
            if (Path(src) / name).exists():
                shutil.copy(Path(src) / name, out / name)
    else:
        from huggingface_hub import hf_hub_download

        for name in TOKENIZER_FILES:
            try:
                shutil.copy(hf_hub_download(src, name), out / name)
            except Exception:  # optional file not in this repo
                pass
    if not (out / "tokenizer.json").exists():
        raise SystemExit(f"{src} has no tokenizer.json; pick models with a fast tokenizer.")


def load(name_or_model):
    if isinstance(name_or_model, torch.nn.Module):
        model = name_or_model
    else:
        model = AutoModelForCausalLM.from_pretrained(name_or_model, dtype=torch.float32)
    # Plain attention exports most reliably to ONNX.
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"
    return model.eval()


def export(observer, performer, out: Path, tokenizer_src: str | None = None,
           names: tuple[str, str] | None = None, quantize: bool = True) -> Path:
    """Export the pair. `observer`/`performer` may be model ids or loaded models (tests)."""
    out.mkdir(parents=True, exist_ok=True)
    tok_src = tokenizer_src or observer
    tok = AutoTokenizer.from_pretrained(tok_src)
    if isinstance(performer, str) and AutoTokenizer.from_pretrained(performer).get_vocab() != tok.get_vocab():
        raise SystemExit("Observer and performer must share a tokenizer.")
    # Copy the original tokenizer files byte for byte (re-saving through
    # transformers can change them). The browser and Python both read these.
    copy_tokenizer_files(tok_src, out)
    if not (out / "config.json").exists():
        # Transformers.js reads config.json next to the tokenizer.
        cfg = observer.config if isinstance(observer, torch.nn.Module) else AutoConfig.from_pretrained(observer)
        cfg.save_pretrained(out)

    model = Binoculars(load(observer), load(performer))
    dummy = torch.tensor([tok("Hello there, this is a short example sentence.")["input_ids"]])
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        # The torch.export-based exporter keeps the sequence length symbolic;
        # the older tracer bakes the example's length into the graph.
        seq = torch.export.Dim("seq", min=2, max=8192)
        program = torch.onnx.export(
            model, (dummy,), input_names=["input_ids"], output_names=["log_ppl", "x_ppl"],
            dynamic_shapes={"input_ids": {1: seq}}, opset_version=18, dynamo=True,
        )
        program.save(str(fp32), external_data=False)
        final = out / "binoculars.onnx"
        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(fp32), str(final), weight_type=QuantType.QUInt8)
        else:
            shutil.copy(fp32, final)

    import hashlib
    sha = hashlib.sha256(final.read_bytes()).hexdigest()[:16]
    obs_name, perf_name = names or (str(observer), str(performer))
    (out / "meta.json").write_text(json.dumps({
        "observer": obs_name, "performer": perf_name, "quantized": quantize, "sha256": sha,
        "size_mb": round(final.stat().st_size / 1e6, 1),
    }, indent=2))
    print(f"Wrote {final} ({final.stat().st_size / 1e6:.0f} MB)")
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", default=os.environ.get("OBSERVER_MODEL") or DEFAULT_OBSERVER)
    ap.add_argument("--performer", default=os.environ.get("PERFORMER_MODEL") or DEFAULT_PERFORMER)
    ap.add_argument("--out", default="site/models")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    export(args.observer, args.performer, Path(args.out), quantize=not args.no_quantize,
           names=(args.observer, args.performer))


if __name__ == "__main__":
    main()
