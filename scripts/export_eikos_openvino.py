#!/usr/bin/env python3
"""Convert the Hugging Face Eikos-4B model to OpenVINO (weight-only INT4 by default).

Model card: https://huggingface.co/caiovicentino1/Eikos-4B  (download that repo into a folder
first, e.g. `huggingface-cli download caiovicentino1/Eikos-4B --local-dir $EIKOS_DIR/Eikos-4B`).

pipeline/label_with_eikos.py expects this layout under $EIKOS_DIR:
    Eikos-4B/            the downloaded Hugging Face folder (tokenizer, decision_core.py)
    eikos-4b-int4-ov/    the output of this script (--mode int4)

Modes: int4 (sym, group size 128, rotary_emb excluded), int8, fp16 (no quantisation).
The export also writes the OpenVINO tokenizer / detokenizer next to the model.

KNOWN GAP: this path is OpenVINO-only (Intel NPU / iGPU). There is no CUDA/ROCm path yet.
Needs: pip install openvino openvino-tokenizers "optimum[openvino]" transformers (a large
download and a memory-hungry conversion; run it once, it is not part of `make all`).

usage: python scripts/export_eikos_openvino.py --src <hf-folder> [--dst <out-folder>] [--mode int4]
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eikos = os.environ.get("EIKOS_DIR")
    ap.add_argument("--src", type=Path, default=Path(eikos) / "Eikos-4B" if eikos else None,
                    help="downloaded Hugging Face Eikos-4B folder (default: $EIKOS_DIR/Eikos-4B)")
    ap.add_argument("--mode", choices=["int4", "int8", "fp16"], default="int4")
    ap.add_argument("--dst", type=Path, default=None,
                    help="output folder (default: $EIKOS_DIR/eikos-4b-<mode>-ov)")
    args = ap.parse_args()
    if args.src is None:
        sys.exit("Pass --src (downloaded Eikos-4B folder) or set EIKOS_DIR.")
    if args.dst is None:
        if not eikos:
            sys.exit("Pass --dst or set EIKOS_DIR.")
        args.dst = Path(eikos) / f"eikos-4b-{args.mode}-ov"

    from openvino import save_model
    from openvino_tokenizers import convert_tokenizer
    from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig
    from transformers import AutoTokenizer

    ignored = {"patterns": [".*rotary_emb.*"]}
    if args.mode == "int4":
        qc = OVWeightQuantizationConfig(bits=4, sym=True, group_size=128, ignored_scope=ignored)
    elif args.mode == "int8":
        qc = OVWeightQuantizationConfig(bits=8, ignored_scope=ignored)
    else:
        qc = None

    src, out = str(args.src), str(args.dst)
    t0 = time.perf_counter()
    model = OVModelForVisualCausalLM.from_pretrained(src, export=True, quantization_config=qc, load_in_8bit=False)
    model.save_pretrained(out)
    print(f"exported {args.mode} in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)

    tok = AutoTokenizer.from_pretrained(src)
    tok.save_pretrained(out)
    ov_tok, ov_detok = convert_tokenizer(tok, with_detokenizer=True)
    save_model(ov_tok, f"{out}/openvino_tokenizer.xml")
    save_model(ov_detok, f"{out}/openvino_detokenizer.xml")
    print("tokenizer exported", flush=True)


if __name__ == "__main__":
    main()
