# NPU / OpenVINO setup

Only the rescue pass (see [METHODOLOGY.md](METHODOLOGY.md), section 3.2) and `make embed-npu` need this.
The rest of the pipeline runs on CPU.

> **Known gap.** This path is OpenVINO-only (Intel NPU for embeddings, Intel iGPU for the LLM). There is no
> CUDA or ROCm alternative in this repository, and a CPU fallback has not been tested. On other hardware the
> rescue pass cannot be run as is; use `make split RESCUE=0` and the rule-only corpus.

Tested on an Intel Core Ultra 7 355 (Panther Lake NPU, PCI `8086:b03e`) with Fedora 44. Other silicon or
distributions need different system packages (step 1); the Python part (step 2) is the same.

## 1. System prerequisites (once per machine; sudo and a reboot)

```bash
sudo dnf install -y intel-npu-driver intel-npu-compiler intel-level-zero
sudo usermod -aG render "$USER"
sudo reboot            # a re-login is not enough for the group change
```

Check afterwards:

```bash
ls /dev/accel/         # accel0
groups                 # includes "render"
```

### Known bug: version-skewed NPU compiler package

On the tested Fedora release, NPU compilation fails for every model, even a one-op graph:

```
Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe
[NPU_DRV] Driver reports a failure from vclAllocatedExecutableCreate2, return code: 2147483646
```

Cause: the distribution ships the driver and the compiler as two independently versioned packages. The
installed compiler exports only `vclAllocatedExecutableCreate` (v1) while the driver calls
`vclAllocatedExecutableCreate2` (v2). Confirm by re-running with `ZE_INTEL_NPU_LOGLEVEL=INFO` and looking
for `Missing symbol function` just before the failure.

Workaround (no root, system package untouched): download the upstream release that matches your installed
driver version (`rpm -q intel-npu-driver`) from the `intel/linux-npu-driver` GitHub releases, extract only
`libnpu_driver_compiler.so` from the `intel-driver-compiler-npu` package inside it (`ar x` the `.deb`, then
untar `data.tar.*`), and put it in a directory of your choice. Then set in `.env`:

```bash
NPU_LIB=/path/to/dir/containing/libnpu_driver_compiler.so
```

The Makefile prepends `NPU_LIB` to `LD_LIBRARY_PATH` for NPU jobs. Set it only if you hit this bug; leave it
empty when the packaged compiler works. A device appearing in `available_devices` does not prove that
compilation works; only the compile step does.

If your login session predates the `render` group change, run NPU commands through `sg render -c "..."`.

## 2. Python environment

```bash
make setup-openvino      # creates .venv-openvino from requirements-openvino.txt
```

This installs CPU-only torch first (the default PyPI torch pulls several GB of NVIDIA libraries), then
OpenVINO, OpenVINO GenAI and optimum-intel, with `transformers` pinned to 4.57.6 (early 5.x releases break
SentencePiece tokenizer conversion). Do not downgrade `optimum`; the `optimum-cli export openvino` command
conflicts with optimum-intel's, which is why export goes through the Python API.

Sanity check:

```bash
.venv-openvino/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"   # ['CPU', 'GPU', 'NPU']
```

Point `NPU_PY` at this environment's python if it lives elsewhere.

## 3. Eikos-4B

Eikos-4B is the LLM that labels papers in the rescue pass. Download it from Hugging Face
(`caiovicentino1/Eikos-4B`) and convert it to an OpenVINO folder:

```bash
.venv-openvino/bin/python scripts/export_eikos_openvino.py    # see the script's --help for options
```

The default output is `models/eikos` (git-ignored), which is also the default `EIKOS_DIR`. Set `EIKOS_DIR`
in `.env` to use another location.

## 4. Which device does what

| Device | Used for | Why |
| --- | --- | --- |
| NPU (`EMBED_DEVICE`, default `NPU`) | bge-large embeddings (single-pass encoder) | Static-shape encoders batch well on the NPU. |
| iGPU (`LABEL_DEVICE`, default `GPU`) | Eikos-4B labelling (autoregressive LLM) | The NPU LLM pipeline cannot batch prefill (batch size 1 only) and decode is slower than CPU/iGPU. |

NPU compilation needs static shapes, so embeddings use three length buckets (128, 256, 512 tokens). Compiled
blobs are cached in `pipeline/caches/ov_npu_blobs/` and by the driver; a first compile of a new shape is
slow, later runs are instant. Models with data-dependent shapes (LSTM/GRU layers, `nonzero`, dynamic
scatter) cannot run on the NPU.

## 5. Expected speeds

| Task | Speed |
| --- | --- |
| Embedding, 128-token bucket | about 90 texts/s |
| Embedding, 256-token bucket | about 45 texts/s |
| Embedding, 512-token bucket | about 12 texts/s |
| Eikos labelling on the iGPU | about 0.5 s per paper |

Embeddings match CPU fp32 at cosine similarity 1.00000. Overall, the rescue pass takes about 2 hours
(METHODOLOGY section 6).

## 6. Memory

| Stage | Peak |
| --- | --- |
| Model conversion (with `requires_grad_(False)` before tracing) | about 3 GB (8.7 GB without it) |
| One-time 512-token NPU compile | about 6.5 GB |

The NPU and iGPU share system RAM. Stages run as separate processes under `MEM_CAP` (default 7G, applied with
`systemd-run` when available); raise it if a stage is killed.

## Known gap (repeated)

OpenVINO only. No CUDA or ROCm path, and the CPU fallback is untested.
