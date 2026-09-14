# LaMa Inpainting Worker — Pure-Rust OxiONNX Backend

**A sidecar worker for the GIMP LaMa inpainting plug-in that runs the model
entirely in Rust — no C/C++, no ONNX Runtime, no protobuf — and is _1.9x
faster than the ONNX Runtime worker_ while producing bit-identical 8-bit
output.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg)](https://www.rust-lang.org)
[![Platform: Windows x64](https://img.shields.io/badge/platform-Windows%20x64-lightgrey.svg)]()
[![Engine: OxiONNX 0.1.7 (vendored)](https://img.shields.io/badge/engine-OxiONNX%200.1.7%20(vendored)-green.svg)](vendor/oxionnx)
[![Speed: 1.92x faster than ORT](https://img.shields.io/badge/speed-1.92x%20vs%20ONNX%20Runtime-brightgreen.svg)](#-benchmarks)
[![Binary: 4.7 MB](https://img.shields.io/badge/binary-4.7%20MB%20(vs%2024.3%20MB)-blueviolet.svg)](#-binary-size)
[![Validation: 472 tests](https://img.shields.io/badge/tests-472%20passing-success.svg)](vendor/oxionnx)
[![Output: bit-identical](https://img.shields.io/badge/output-bit--identical-success.svg)](#-validation)

---

## Highlights

- **1.92x faster than ONNX Runtime** on the same machine, same model
  (7.57 s → 3.95 s for 512×512, six interleaved A/B rounds).
- **4.7 MB binary** versus ORT's 24.3 MB — a 5.2x smaller sidecar, with zero
  native dependencies to ship.
- **Bit-identical output**: every optimisation in this repository is checked
  against the original engine's PNG output byte-for-byte on three reference
  images.
- **12.3x faster than stock OxiONNX 0.1.7** (48.3 s → 3.93 s) — the speed came
  from profiling and fixing a handful of pathological code paths, not from
  swapping engines (see [Benchmarks](#-benchmarks) and
  [docs/OXIONNX_REPORT.md](docs/OXIONNX_REPORT.md)).
- **The engine is vendored in-tree** (`vendor/oxionnx/`, Apache-2.0), with a
  complete fork record in [MODIFICATIONS.md](vendor/oxionnx/MODIFICATIONS.md):
  every change, its measured impact, and the two attempts that were measured
  *slower* and reverted.

## How it fits together

```
GIMP 3.2 process (MINGW Python + GEGL)
  └─ lama-inpaint.py  …… exports drawable + mask to temp PNGs
        └─ spawns a worker process (out of process, always)
              ├─ lama_worker.py        Python + onnxruntime   (default)
              ├─ lama-worker.exe       Rust + ONNX Runtime    (opt-in)
              └─ lama-worker-oxionnx   Rust + OxiONNX   ←──  this repository
                    · loads lama_fp32.onnx (17,480 nodes)
                    · pads to mod-16, one inference, soft-mask composite
                    · writes RGBA result with the input alpha preserved
```

The worker is a **drop-in replacement**: identical CLI
(`--image --mask --model --output`), identical markers on stderr
(`[LAMA_MARKER] phase …`), identical PNG conventions. The GIMP side does not
need to know which engine ran.

## 📊 Benchmarks

### The headline: vs ONNX Runtime

Six interleaved rounds in the same minutes (this laptop's thermal state moves
results by ±20%, so alternating measurements are the only fair comparison):

| Round | ONNX Runtime (CPU) | OxiONNX worker |
|---|---|---|
| 0 | 7.54 s | **3.94 s** |
| 1 | 7.45 s | **3.93 s** |
| 2 | 7.61 s | **3.95 s** |
| 3 | 7.62 s | **3.96 s** |
| 4 | 7.63 s | **3.93 s** |
| 5 | 7.55 s | **3.96 s** |
| **Average** | **7.57 s** | **3.95 s** |

`512×512 input · Intel i7-13620H · Rust 1.94 · same model file`

### Binary size

| Worker | Size | Dependencies |
|---|---|---|
| ONNX Runtime worker | 24.3 MB | `onnxruntime` shared library + protobuf |
| **OxiONNX worker** | **4.7 MB** | none — pure Rust, statically linked |

### The optimisation arc

Every step below is a measured node-time change, verified bit-identical:

| Stage | 512×512 total | What landed |
|---|---|---|
| Stock OxiONNX 0.1.7 | 48.3 s | — |
| + ConvTranspose algorithm rewrite | 15.5 s | direct output-oriented kernel, rayon |
| + AVX2 kernels & Pad fast path | 11.9 s | `vgatherdps` asm, NCHW memcpy-pad |
| + GEMM fixes | 9.0 s | N-split packing, merged batched GEMM, direct small-M conv |
| + shape-op & SIMD wave | 4.5 s | Slice, Div, Transpose, Einsum, 2-channel ConvTranspose, BatchNorm |
| + session cache & ORT-inspired fusions | **3.93 s** | cache keyed by revision; two graph folds |

### What didn't work (kept on the record)

| Attempt | Why it seemed right | Measurement |
|---|---|---|
| Custom small-K AVX2 GEMM for `m=64 k=64` Einsum | `matrixmultiply` never amortises its packing over a 64-wide K | **Slower**: 1,101 ms vs 352 ms — the broadcast-per-FMA shape is port-5-bound |
| Row-block parallel ConvTranspose | Keep each task's input slice cache-resident | **Slower**: 387 ms vs 354 ms — the input already lives in L3 across fine-grained tasks |
| AVX2 intrinsics gather kernel (pre-assembly) | Vectorise the channel dot product | Superseded: the inline-asm `vgatherdps` version beat it, then **both** were deleted by the row-wise kernel that removed gathers entirely (2.4x faster still) |

### Where the time goes now

| Op | Time | Note |
|---|---|---|
| Conv (222 nodes) | ~1,280 ms | im2col + `matrixmultiply`, at its parallel ceiling |
| Einsum (216) | 355 ms | N-split parallel |
| Transpose (566) | 250 ms | block-swap decomposition + AVX2 tiles |
| Slice (1,322) | 212 ms | bulk-copy fast path |
| Reshape + Concat (2,708) | ~425 ms | memcpy floor |
| ConvTranspose (3) | 205 ms | row-wise + 2-channel AVX2 |
| MatMul (216) | ~200 ms | merged-GEMM broadcast folding |
| everything else | ~600 ms | Pad, Div, BatchNorm, Gather, … |

## 🗺️ Milestone history

| Milestone | Outcome |
|---|---|
| **M1 — Can it run at all?** | OxiONNX 0.1.4 failed on LaMa: `Pad` panic → reported with repro + patch |
| **M2 — Upstream fix** | Maintainer found the real cause (negative `Slice` steps clamped) and shipped 0.1.5; LaMa runs E2E at 48.3 s |
| **M3 — First evaluation** | Binary 5.5x smaller, output bit-identical outside the mask, CUDA tested (correct but slower — GPU round trips), speed 5.5x behind ORT. [Full report](docs/OXIONNX_REPORT.md) posted upstream |
| **M4 — Profile & fix the 75%** | Three ConvTranspose nodes were 75% of runtime → direct kernel + AVX2 → 15.5 s |
| **M5 — Systematic campaign** | 13 fixes across conv/GEMM/shape ops, each profiled and A/B verified → 3.93 s |
| **M6 — Pass ORT** | Interleaved A/B: **1.92x faster than ONNX Runtime**, 4.7 MB vs 24.3 MB |
| **M7 — Harden & vendor** | Engine vendored with Apache-2.0 notices, 472 unit tests, cache revisioning, ORT strategy audit ([docs/ORT_STRATEGIES.md](docs/ORT_STRATEGIES.md)) |
| **M8 — Next** | GIMP plug-in wiring + large-image ROI path (see [Roadmap](#-roadmap)) |

## 🧪 Validation

- **472 unit tests** in the vendored engine (`cargo test --release --features simd --lib`).
- **Bit-identical output** against the original engine on three reference
  images (512×512 synthetic, 800×600 photo-like, 1000×700 mod-16 edge case):
  max per-channel difference **0**.
- **Interleaved A/B measurement** for every performance claim, with
  kill switches (`OXIONNX_NO_ADDBN_FUSION`, `OXIONNX_NO_SCRATCH_CACHE`,
  `OXIONNX_NO_SESSION_CACHE`) so any optimisation can be turned off and
  re-measured in the same binary.
- **Session-cache integrity**: cache file names embed a revision plus the
  model's size/mtime, and are written atomically — a stale or truncated cache
  is never loaded.

## 🧭 Roadmap

- [ ] **Large-image ROI path** — above 4 MP, switch to
      bbox → context-pad → crop, as the ORT/Python workers do; today the
      worker runs the whole image at native resolution.
- [ ] **GIMP plug-in wiring** — installer, `worker_kind: "oxionnx"` in
      `lama_config.json`, model + cache placement under the GIMP profile.
- [ ] **Worker-level integration tests** — spawn the binary, assert
      RGBA/alpha/size behaviour, mirroring the ORT worker's
      `tests/integration.rs`.
- [ ] **v2 engine work** — implicit-GEMM convolution (removes the 19–56 MB
      im2col matrices) and a stride-based `Tensor` (zero-copy reshape).
      See [docs/OXIONNX_REPORT.md §9](docs/OXIONNX_REPORT.md).

## 📁 Repository layout

```
.
├── Cargo.toml / Cargo.lock     worker crate (path-deps vendor/oxionnx)
├── LICENSE                     Apache-2.0
├── src/
│   ├── main.rs                 the worker: CLI, image IO, pre/post, cache
│   └── bin/bench_gemm.rs       GEMM microbenchmark used during profiling
├── vendor/oxionnx/             vendored OxiONNX 0.1.7 (Apache-2.0)
│   └── MODIFICATIONS.md        the fork record: changes, impacts, reverts
├── docs/
│   ├── OXIONNX_REPORT.md       evaluation + optimisation report (the long read)
│   └── ORT_STRATEGIES.md       audit of onnxruntime-main: adopted / rejected / v2
└── test_data/                  six input fixtures (generated outputs gitignored)
```

## 🔨 Build & run

```bash
cargo build --release
# produces: target/release/lama-worker-oxionnx.exe

./target/release/lama-worker-oxionnx.exe \
    --image input.png --mask mask.png \
    --model lama_fp32.onnx --output result.png
```

The engine's `simd` feature is **required** (the fork's AVX2 kernels are
`unsafe`; the crate denies `unsafe_code` without it) and is enabled by the
worker's dependency declaration. AVX2/FMA are runtime-detected; scalar
fallbacks exist for other x86-64 CPUs.

Optional environment switches, all off by default:

| Variable | Effect |
|---|---|
| `OXIONNX_PROFILE=1` | per-node timing dump (top 50 + totals by op) and `profile_all_nodes.csv` |
| `OXIONNX_NO_SESSION_CACHE=1` | disable the `save_optimized` session cache |
| `OXIONNX_NO_ADDBN_FUSION=1` | disable the `fuse_add_batchnorm` graph fold |
| `OXIONNX_NO_SCRATCH_CACHE=1` | allocate a fresh im2col buffer per convolution |

## 🤝 Acknowledgements

- **[OxiONNX](https://github.com/cool-japan/oxionnx)** — the pure-Rust engine
  this work builds on. The upstream maintainer diagnosed our 0.1.4 report to
  its real root cause (`Slice` clamping) and shipped the fix in 0.1.5,
  including a legacy-opset `Pad` repair for 0.1.8.
- **ONNX Runtime** — the performance baseline, and the source of two graph
  fusions adopted here (`conv_bn_fusion`, `conv_add_fusion`); see
  [docs/ORT_STRATEGIES.md](docs/ORT_STRATEGIES.md).
- **`matrixmultiply`** — the pure-Rust GEMM underneath every convolution.

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE). The vendored engine in
`vendor/oxionnx/` is Apache-2.0 from the OxiONNX project, with this
repository's modifications marked per file and summarised in
[vendor/oxionnx/MODIFICATIONS.md](vendor/oxionnx/MODIFICATIONS.md).
