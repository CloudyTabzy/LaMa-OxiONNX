# Vendored OxiONNX — fork record

This directory is a vendored copy of **OxiONNX 0.1.7**
(<https://github.com/cool-japan/oxionnx>, Apache-2.0 — see `LICENSE`) carrying
this project's performance patches for LaMa inpainting inference. It is
consumed by the sibling worker crate as a path dependency
(`oxionnx = { path = "vendor/oxionnx", features = ["simd"] }`).

Every modified file carries a leading comment describing its change
(Apache-2.0 §4(b)). This document is the full record.

## Build requirement

The **`simd` feature is mandatory** for this fork, not optional. The added
AVX2 kernels are `unsafe`, and the crate is
`#![cfg_attr(not(feature = "simd"), deny(unsafe_code))]`, so a build without
it fails to compile. The worker's `Cargo.toml` enables it unconditionally.

## Why the fork

The upstream 0.1.7 engine ran LaMa in ~48 s on the reference machine. The
upstream issue (<https://github.com/cool-japan/oxionnx/issues/4>) documented
the model-loading failures we found at 0.1.4 and the upstream fixes; after
those landed we profiled per-node and patched the hot paths. The fork now
runs the same graph at ~3.9 s — faster than ONNX Runtime's CPU path
(7.5 s) on the same machine — with bit-identical 8-bit output.

## Changes

Ordered as they were made. Measurements are `OXIONNX_PROFILE=1` node totals
on the 512×512 LaMa graph, on an i7-13620H.

### `oxionnx-ops`

| File | Change | Impact |
|---|---|---|
| `src/conv/transpose.rs` | Row-wise AVX2 ConvTranspose kernels for stride 2 / kernel 3 (single- and two-output-channel), replacing a per-pixel `vgatherdps` kernel and, before that, a scalar scatter-accumulate | 27 967 → 205 ms (3 nodes) |
| `src/conv/conv2d.rs` | Direct small-M conv kernel for `c_out ≤ 3` (no im2col); N-split parallel GEMM so each thread packs B once instead of every thread packing all of it; thread-local reusable im2col scratch buffer | small-M conv 909 → 66 ms; conv total 2 894 → 1 410 ms |
| `src/shape/basic.rs` | Block-swap transpose decomposition covering every permutation in the graph; AVX2 8×8 register-tile transpose; `[P,2]↔[2,P]` interleave/deinterleave kernels; tile-order heuristic for `Q ≫ P` | 635 → 250 ms |
| `src/shape/sequence.rs` | Bulk-copy fast path in `slice()` keyed on the last sliced axis (was a per-element division walk) | 325 → 207 ms |
| `src/math/broadcast.rs` | Scalar-divisor path in `div()` using IEEE broadcast `vdivps` (every `Div` in LaMa is `x / const`) | 251 → 153 ms |
| `src/math_typed.rs` | Merged-GEMM broadcast folding in `matmul_f32_into`: `b_batches == 1` stacks A along M; `a_batches == 1, n == 1` computes `C' = B @ Aᵀ` | batched MatMul 1 792 → 330 ms (98 446 tiny GEMM calls → ~400) |
| `src/einsum/contract.rs` | N-split parallel path for single-batch contractions (`m=64, k=64, n=6336\|12288` ran single-threaded) | 409 → 342 ms |
| `src/nn/normalization.rs` | Single-pass SIMD batch-norm channel kernel (was a scalar loop plus a full `copy_from_slice`) | 196 → 107 ms |
| `src/simd_ops/{avx2,functions,mod}.rs` | `div_scalar` and `batch_norm_channel` AVX2 kernels and their dispatch wrappers | (support for the three rows above) |

### `src/optimizer` (graph-level, ORT-inspired)

| File | Change | Impact |
|---|---|---|
| `src/optimizer/fusion/conv/add_batchnorm.rs` (**new**) | `fuse_add_batchnorm`: folds `BatchNorm(Add(conv_a, conv_b))` — the FFC block shape — into both branches' weights, carrying the whole shift on one branch's bias. ONNX Runtime reaches the same graph shape in two passes (`conv_add_fusion` + `conv_bn_fusion`); since both branches are Convs here, one pass suffices | 72 BatchNorm nodes removed; −53 ms measured A/B |
| `src/optimizer/fusion/conv/batchnorm.rs` | `fuse_conv_batchnorm` extended to `ConvTranspose` (output channel is weight dim 1, blocks are `kH*kW` runs strided by `C_out`) | ConvTranspose 205 → 167 ms |
| `src/optimizer/fusion/conv/tests.rs` | Unit tests for the new pass (fold, shared-operand decline, graph-output decline) | 472 tests total, all green |
| `src/optimizer/{mod,fusion/mod,fusion/conv/mod}.rs` | Pass wiring, exports, and the `OXIONNX_NO_ADDBN_FUSION` kill switch used to measure the fold A/B in one binary | — |

### Measured and reverted (documented so they are not retried)

* **Custom small-K AVX2 GEMM** for the Einsum contractions (serial and
  M-split-parallel): both slower than the parallel `matrixmultiply` path
  (1 101 ms vs 352 ms). The broadcast-per-FMA structure is port-5-bound.
* **Row-block parallel ConvTranspose** (one task = rows × all channels):
  387 ms vs 354 ms — on this CPU the input already stays resident across the
  fine-grained per-plane tasks, so coarse tasks only cost load balance.

## Upgrading upstream

When rebasing onto a newer OxiONNX:

1. Re-apply the patches listed above (the file-level notices mark every
   site).
2. Re-run `cargo test --release --features simd --lib` from this directory.
3. Re-run the worker's image comparisons (three reference images in
   `../test_data/`, all must be bit-identical) — and **bump the worker's
   `SESSION_CACHE_REVISION`** so stale caches from the previous optimizer are
   not loaded.

## Notes for upstream

Everything here is offered back to the OxiONNX project; the issue thread has
the profiling data and the upstream fixes that unblocked LaMa. See
`../../docs/OXIONNX_REPORT.md` in the worker repository for the write-up.
