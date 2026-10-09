#!/usr/bin/env python3
"""Label papers CORE / PERIPHERAL / NOT with Eikos-4B (single forward pass, letter read-out).

Eikos-4B is a fine-tuned Qwen3.5-4B that answers a *typed decision* by reading the probability
of each option letter at the first answer position (no text is generated), so one paper costs
one prompt prefill. The rubric is pipeline/rescue_rubric.py, identical to the one given to the
Haiku and Sonnet audit sub-agents. Eikos was trained on finance and rules decisions, so using
it for paper triage is out of its home domain; docs/METHODOLOGY.md reports how well it agrees
with the Haiku and Sonnet labels on the same papers (about 81 % with Haiku, kappa 0.71).

Runs through OpenVINO on the Intel iGPU (`GPU`): measured ~0.6 s per paper, ~3.5 GB, far better
than the NPU, which handles single-pass encoders well but not LLM prefill. Resumable: finished
items are skipped when the output file already exists.

Input  JSONL: one object per line with at least {item, title, abstract}; optional {year}.
Output JSON : [{item, label, probs, tokens}], probs ordered as rescue_rubric.LABELS.

Needs an environment with openvino + transformers (`make setup-openvino`) and the exported model:
    python label_with_eikos.py --in corpus/data/rescue_label_sample.jsonl \\
        --out corpus/data/rescue_label_sample_eikos.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rescue_rubric import LABELS, QUESTION  # noqa: E402

DEFAULT_EIKOS_DIR = os.environ.get("EIKOS_DIR") or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="GPU", help="OpenVINO device for the language model")
    ap.add_argument("--eikos-dir", type=Path, default=DEFAULT_EIKOS_DIR,
                    help="(default: env EIKOS_DIR) folder holding Eikos-4B/ (tokenizer, decision_core.py) and the exported model")
    ap.add_argument("--model", default="eikos-4b-int4-ov", help="exported OpenVINO model folder name")
    args = ap.parse_args()
    if not args.eikos_dir:
        sys.exit("Eikos model folder not set: pass --eikos-dir or set the EIKOS_DIR environment variable "
                 "(e.g. in .env). Create the model with scripts/export_eikos_openvino.py.")

    # The decision format and read-out are shared with the model's training code, so the model
    # answers exactly as it was trained (decision_core.py ships with the model).
    os.environ["PROMPT_STYLE"] = "semif"
    sys.path.insert(0, str(args.eikos_dir / "Eikos-4B"))
    import openvino as ov
    from decision_core import LABELS as LETTERS
    from decision_core import messages, options_of
    from transformers import AutoTokenizer

    items = [json.loads(line) for line in open(args.inp, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(str(args.eikos_dir / "Eikos-4B"))
    mdir = args.eikos_dir / args.model
    core = ov.Core()
    emb = core.compile_model(core.read_model(str(mdir / "openvino_text_embeddings_model.xml")), "CPU")
    t0 = time.perf_counter()
    cm = core.compile_model(core.read_model(str(mdir / "openvino_language_model.xml")), args.device)
    print(f"compiled on {args.device} in {time.perf_counter() - t0:.0f}s", flush=True)
    req = cm.create_infer_request()

    opts = options_of(QUESTION)
    assert tuple(o[0] for o in opts) == LABELS, "option order must match rescue_rubric.LABELS"
    letter_ids = [tok(LETTERS[i], add_special_tokens=False)["input_ids"][0] for i in range(len(opts))]

    res = json.load(open(args.out)) if args.out.exists() else []
    done = {x["item"] for x in res}
    t0 = time.perf_counter()
    for k, it in enumerate(items):
        if it["item"] in done:
            continue
        state = {"year": it.get("year"), "title": it["title"], "abstract": it["abstract"]}
        text = tok.apply_chat_template(messages(state, QUESTION, opts), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="np", add_special_tokens=False)["input_ids"]
        n = ids.shape[1]
        feed = {"attention_mask": np.ones((1, n), np.int64), "inputs_embeds": emb(ids)[0].astype(np.float32),
                "position_ids": np.broadcast_to(np.arange(n, dtype=np.int64), (4, 1, n)).copy(),
                "beam_idx": np.zeros(1, np.int32)}
        req.reset_state()
        req.infer(feed)
        logits = req.get_tensor("logits").data[0, -1]
        sel = np.array([logits[i] for i in letter_ids], np.float64)
        p = np.exp(sel - sel.max())
        p /= p.sum()
        res.append({"item": it["item"], "label": LABELS[int(p.argmax())],
                    "probs": [round(float(x), 4) for x in p], "tokens": n})
        if (k + 1) % 50 == 0:
            tmp = args.out.with_suffix(".tmp")
            json.dump(res, open(tmp, "w"))
            os.replace(tmp, args.out)
            print(f"{k + 1}/{len(items)}  {(time.perf_counter() - t0) / max(1, len(res) - len(done)):.2f}s/paper", flush=True)
    json.dump(res, open(args.out, "w"))
    print(f"DONE {len(res)} items -> {args.out}")


if __name__ == "__main__":  # without this guard a spawned worker re-runs the whole script
    main()
