"""End-to-end check of the model conversion with a tiny randomly initialised model.

Verifies that the exported ONNX graph reproduces the PyTorch Binoculars maths,
and (when the JS libraries are installed, see JS_DEPS below) that the browser
code produces the same scores from the same file. Skipped in CI, where torch
is not installed.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from detector.lm_scores import BinocularsScorer  # noqa: E402
from scripts.export_models import Binoculars, export  # noqa: E402

SITE_JS = Path(__file__).resolve().parent.parent / "site" / "js"
# Directory containing node_modules with onnxruntime-web and @huggingface/transformers.
JS_DEPS = os.environ.get("JS_DEPS")

TEXT = ("The committee met on Tuesday to review the budget. Several members raised concerns about "
        "rising costs, while others argued the plan was sound. A final vote is expected next month.")


def make_tokenizer(tmp: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([TEXT] * 20 + ["hello world, café naïve résumé"] * 5,
                            trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                special_tokens=["<|endoftext|>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", clean_up_tokenization_spaces=False)
    fast.save_pretrained(tmp / "tok")
    return fast


def tiny_llama(vocab, seed):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    tok = make_tokenizer(tmp)
    obs, perf = tiny_llama(len(tok), 1), tiny_llama(len(tok), 2)
    out = tmp / "models"
    export(obs, perf, out, tokenizer_src=str(tmp / "tok"), names=("tiny-obs", "tiny-perf"), quantize=False)
    return out, tok, obs, perf


def test_onnx_matches_pytorch(exported):
    out, tok, obs, perf = exported
    from tokenizers import Tokenizer

    # The shipped tokenizer.json is what both the browser and Python use.
    ids = Tokenizer.from_file(str(out / "tokenizer.json")).encode(TEXT, add_special_tokens=False).ids
    with torch.no_grad():
        lp_ref, xp_ref = Binoculars(obs, perf)(torch.tensor([ids]))
    scores = BinocularsScorer(out).score(TEXT)
    # Must differ from the export's example length, to prove the length isn't baked in.
    assert len(ids) != 11
    assert len(scores.log_ppl) == len(ids) - 1
    np.testing.assert_allclose(scores.log_ppl, lp_ref.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(scores.x_ppl, xp_ref.numpy(), rtol=1e-4, atol=1e-4)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["observer"] == "tiny-obs" and meta["sha256"]


def test_quantized_export_runs(tmp_path, exported):
    _, tok, obs, perf = exported
    out = tmp_path / "q"
    export(obs, perf, out, tokenizer_src=str(exported[0]), names=("a", "b"), quantize=True)
    s = BinocularsScorer(out).score(TEXT).summary()
    assert np.isfinite(s["binoculars"])


@pytest.mark.skipif(not JS_DEPS, reason="set JS_DEPS to a folder with onnxruntime-web + @huggingface/transformers")
def test_browser_code_matches_python(exported):
    out, *_ = exported
    py = BinocularsScorer(out).score(TEXT)
    script = f"""
      import * as ort from "onnxruntime-web";
      import {{ AutoTokenizer, env }} from "@huggingface/transformers";
      import {{ Detector }} from "{(SITE_JS / 'detector.js').as_uri()}";
      import {{ readFileSync }} from "node:fs";
      env.allowRemoteModels = false; env.localModelPath = "{out.parent}/";
      const tokenizer = await AutoTokenizer.from_pretrained("{out.name}");
      const session = await ort.InferenceSession.create(new Uint8Array(readFileSync("{out}/binoculars.onnx")));
      const det = new Detector({{ ort, session, tokenizer, classifier: null }});
      const s = await det.score({json.dumps(TEXT)});
      console.log(JSON.stringify({{ lp: Array.from(s.lp), xp: Array.from(s.xp), offsets: s.offsets }}));
    """
    # Run from a file inside JS_DEPS so bare imports resolve (and worker threads
    # don't inherit an --eval flag).
    script_path = Path(JS_DEPS) / "parity_check.mjs"
    script_path.write_text(script)
    res = subprocess.run(["node", str(script_path)], cwd=JS_DEPS, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout.strip().splitlines()[-1])
    np.testing.assert_allclose(js["lp"], py.log_ppl, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(js["xp"], py.x_ppl, rtol=1e-4, atol=1e-4)
    assert [tuple(o) for o in js["offsets"]] == py.offsets
