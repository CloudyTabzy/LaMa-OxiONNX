# ONNX Runtime strategies, audited against the LaMa worker

A read of `C:\Dev\GIMP_Native_Plugin\onnxruntime-main` hunting for techniques
that apply to our vendored-OxiONNX worker, with a verdict per technique:
**adopted**, **already had it**, **rejected** (with the measurement that
rejected it), or **v2**. File references are to the ORT tree; "ours" means
`vendor/oxionnx` or the worker.

## 1. MLAS — the SGEMM organisation

What ORT does (`onnxruntime/core/mlas/lib/sgemm.cpp`,
`.../lib/x86_64/FgemmKernelFma3Common.h`):

| Technique | Detail | Our status |
|---|---|---|
| **PanelA/PanelB stack buffers** | `MlasSgemmOperation` packs into fixed-size stack panels (`9,096`: `PanelA[12*128]`, `PanelB[128*128]`, 16-byte aligned). No heap allocation per GEMM. | `matrixmultiply` allocates its pack buffers internally — not reachable from our side. We compensate by *reducing the number of GEMM calls* (merged batched matmul: 98 446 calls → ~400). |
| **Adaptive StrideN/StrideK** | `9,180`: when `N >= K` the blocking doubles StrideN and halves StrideK until K fits one block; when `K > N` it does the reverse. | **v2.** Our small-K Einsum shapes (`m=64, k=64, n=6336`) are exactly what this handles — ORT would run them as one K-block × wide N slices. Our own small-K kernel was tried and measured *slower* than the parallel `matrixmultiply` path (see §5), so this needs a better kernel, not just different blocking. |
| **Split the larger dimension** | `MlasGemmBatch` (`~1560`): `if (N > M) { ThreadCountM=1; ThreadCountN=… } else { ThreadCountM=…; ThreadCountN=1 }`. Duplicate A-packing across M-threads is accepted when M is the smaller axis. | **Adopted in spirit.** `parallel_sgemm` splits N (B packed once) with an M-split fallback below `n = threads*32`. Every hot GEMM in LaMa has N > M, so ORT's rule and ours agree on this workload. |
| **Complexity-scaled thread count** | `MLAS_SGEMM_THREAD_COMPLEXITY = 64*1024` multiplies per thread; small GEMMs run single-threaded. | **Already equivalent.** Our threshold is `PARALLEL_GEMM_FLOPS` + `n >= threads*32`; the Einsum chunk sweep (4/8/16) confirmed one chunk per thread is best at our sizes. |
| **M == 1 → GEMV kernel** | `9,000`: "The data from matrix B is not referenced multiple times, so using a local packed buffer is a wasted memory copy." Dedicated `SgemmKernelM1`. | **Already had it.** Our direct small-M Conv (`c_out ≤ 3`, no im2col) is the same realisation for the LaMa final conv: 909 → 66 ms. |
| **N == 1 → transpose trick** | `9,120`: `Transpose(A·B) = Transpose(B)·Transpose(A)`, reuse the M1 kernel. | **Already had it.** The merged-batch MatMul case (`a_batches==1, n==1`) computes `C' = B @ Aᵀ` with the batch on M. |
| **Fused activation + Sum epilogues** | `providers/cpu/nn/conv.cc ~390`: `activation_` and a `Sum` input are passed into `MlasConvPrepare`/`MlasConv`; the sum becomes the GEMM's `beta = 1`. Zero extra passes over the output. | **Partially.** `fuse_conv_relu` exists and already fires (77 Relu nodes folded). The `Sum` (conv+add) epilogue would need a `Conv` op that reads an accumulator; our `ConvAddRelu` op is referenced by the optimizer but has **no kernel**, so `fuse_conv_add_relu` is dead in our build. See §3 note. |

## 2. Prepacking — the weight-side win ORT gets for free

`providers/cpu/math/matmul.cc:302` `MatMul::PrePack` packs **B** once at
session init (`GemmPackBFp32` → `MlasGemmPackB`), with a
`MlasSgemmPackedOperation` fast path and `PrePackedWeights` sharing across
sessions. Convolution filters get the same treatment where the backend
supports it (`Parameters.PackedFilter`).

**Our status: not applicable, for a specific reason.** In our im2col GEMMs
the *weights* are the **A** operand and the im2col matrix is B; prepacking
helps B (the per-inference activation), not A. The weights are small
(≤ 1.7 MB) and are re-packed per call inside `matrixmultiply`, but that pack
is a small fraction of each call. The measurable version of this idea for us
was already taken in the *opposite* direction: eliminate the im2col B
materialisation (see `ImplicitGemmConv` below), which is §3.

## 3. Convolution algorithm selection

`onnxruntime/core/mlas/lib/convolve.cpp` `MlasConvPrepare` (~1650–1810):

| Rule | ORT behaviour | Ours |
|---|---|---|
| Pointwise (`K == C_in`, stride 1, no pad) | `GemmDirect` — no im2col at all | `conv2d_1x1_into` — same |
| Kernel spans the whole input with `C_in == 1` | `GemmDirect` with `TransB` | not applicable to LaMa |
| `FilterCount > OutputSize` | full `ExpandThenGemm`, working buffer = `OutputSize * K` | our 64 MB-capped im2col, column-blocked |
| otherwise | `ExpandThenGemmSegmented`: **per-thread buffer of `StrideN × StrideK` = 128 × 128 floats = 64 KB**, each thread packs its own N-segment panel and loops K in `StrideK` chunks | **v2 — the most interesting remaining item.** Our im2col builds a 19–56 MB column matrix in DRAM: 56 MB written, then 56 MB read by the GEMM. ORT's 64 KB panels stay in L2, so the same convolution touches DRAM roughly half as much. Porting it means restructuring `conv2d_into_slices` around `(N-segment, K-chunk)` panel packing with a `beta=1` accumulate — not a drop-in change to `matrixmultiply`'s calling pattern, so it needs its own measurements before adoption. |
| Depthwise specialisations | dedicated kernels | not present in LaMa (no `groups > 1` convs; verified) |

**Implicit GEMM** (`ImplicitGemmConv` dispatches in OxiCUDA / ORT's kernels
elsewhere) — building the B panel **from the source image inside the GEMM's
pack loop**, skipping the intermediate matrix entirely — is the same idea one
step further. **v2.**

## 4. Graph fusions (our crate already has most of the machinery)

`onnxruntime/core/optimizer/*` vs our `src/optimizer/fusion/*`:

| ORT pass | Ours | Verdict |
|---|---|---|
| `conv_bn_fusion` | `fuse_conv_batchnorm` | **Extended this session** to `ConvTranspose` (ORT does not cover it; LaMa's decoder has 3 `ConvTranspose+BN`s). Measured: ConvTranspose node total 205 → 167 ms. |
| `conv_add_fusion` + `conv_bn_fusion` | — | **Adopted this session** as `fuse_add_batchnorm`, a single pass for the pattern ORT reaches in two: `BN(Add(conv_a, conv_b))` → scale both convs and carry the whole shift on one branch's bias. LaMa's FFC blocks have 72 of these. Node count 9 900 → 9 828; measured A/B **−53 ms** (`run_ms` 3 654 vs 3 707, 4 interleaved rounds); outputs bit-identical. |
| `conv_activation_fusion` | `fuse_conv_relu`, `fuse_conv_clip_to_conv_relu6` | already firing (77 Relu nodes folded into `activation="relu"` convs) |
| `conv_add_act_fusion` (ResNet) | `fuse_conv_add_relu` | **dead code in our build** — it emits `OpKind::ConvAddRelu`, which has no kernel in the registry. Fixing that is a v2 feature (write the fused op), not hardening. |
| `pad_fusion` (fold Pad into Conv) | — | **not applicable**: LaMa's pads are `reflect`, and ONNX `Conv` padding is zero-fill. (ORT's PadFusion only folds constant/zero pads.) |
| `transpose_optimizer`, `slice_elimination`, `reshape_fusion`, `noop_elimination` | `cancel_consecutive_transpose/reshape`, `simplify_transpose_reshape` | partially applicable; the FFC's shape-op chains are driven by runtime `Shape` values, so constant folding cannot see through them. Leave as v2. |
| `constant_folding` | `constant_fold` + `materialize_shape_ops` | ran; the FFC's `Shape`+`Range`+`Cos/Sin` matrix construction is dynamic-shape and survives, as in ORT without shape overrides. |
| `pre_shape_node_elimination` | — | v2, marginal. |

## 5. Measured experiments this session (both reverted)

* **Custom small-K AVX2 GEMM** for the Einsum contractions
  (`m=64, k=64, n=6336|12288`), serial and M-split-parallel variants.
  matrixmultiply's per-call packing never amortises over a 64-wide K loop, so
  a dedicated kernel *should* win — it did not: the broadcast-per-FMA
  structure (~4 MACs/instruction, port-5-bound) measured **slower** than the
  N-split `matrixmultiply` path (1 101 ms vs 352 ms for the op total on the
  parallel variant). Reverted; the N-split stands.
* **Row-block parallel ConvTranspose** (one task = N rows × all channels, to
  keep the input slice cache-resident). Measured 387 ms vs 354 ms: on this
  CPU (i7-13620H, 24 MB L3) the 8.4 MB input already stays resident across
  the 256 fine-grained plane tasks, so the coarse tasks only cost load
  balance. Reverted.

## 6. Other ORT engineering worth knowing (not adopted)

* **Memory pattern planner** (`framework/mem_pattern_planner.h`): traces the
  allocation/free sequence of one run and replays it for subsequent runs with
  the same input shape. We have a size-class pool; a replayable plan is a
  possible v2 for the ~270 ms of `session.run` scaffolding.
* **ORT format / flatbuffers** (`core/flatbuffers`): a binary model format
  that skips protobuf parsing. Ours is the session cache
  (`save_optimized`/`load_optimized`, ~145 ms load), same idea.
* **Spin-then-block thread pool** (`include/onnxruntime/core/platform/threadpool.h:133`):
  configurable spin duration/backoff for idle workers so back-to-back
  parallel regions don't pay wake-ups. Rayon spins briefly already; a
  calibrated spin (ORT exposes `kSpinDurationDefault`) is worth knowing if we
  ever see scheduler-bound phases.
* **Channels-last fast paths** (`nhwc_fastpath` in `conv.cc`): a layout
  rewrite that lets some convs avoid transposes. LaMa's FFC is NCHW
  throughout; not applicable.

## 7. What was actually changed

1. `src/optimizer/fusion/conv/add_batchnorm.rs` — new `fuse_add_batchnorm`
   pass with 3 unit tests (fold, shared-operand decline, graph-output
   decline). Wired into `optimize_with_input_shapes` after
   `fuse_conv_batchnorm`, with an `OXIONNX_NO_ADDBN_FUSION=1` kill switch for
   A/B measurement (matching our existing `OXIONNX_NO_SCRATCH_CACHE`).
2. `src/optimizer/fusion/conv/batchnorm.rs` — `fuse_conv_batchnorm` extended
   to `ConvTranspose` (output channel is weight dim 1, blocks are `kH*kW`
   runs strided by `C_out`).
3. Docs: this file.

Full test suite: **472 passed, 0 failed** (`cargo test --release --features
simd --lib`). All three reference images bit-identical (512×512, 800×600,
1000×700; max per-channel diff 0).
