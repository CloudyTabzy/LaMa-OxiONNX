# OxiONNX 0.1.7 — LaMa evaluation and optimisation report

This is the consolidated technical report on running **LaMa inpainting**
(`lama_fp32.onnx`, 17,480 nodes, 198 MB) on **OxiONNX 0.1.7** — the pure-Rust
ONNX inference engine — as the backend for the GIMP LaMa worker.

It covers the two phases of the work:

1. **Evaluation** — does the engine run the model at all, how big is the
   binary, how fast is it, and what does the CUDA path do (September 2026).
2. **Optimisation** — a per-node profiling campaign that took the same
   workload from 48.3 s to 3.93 s, ending **1.92x faster than ONNX Runtime's
   CPU path** on the same machine. The output initially differed from ORT in
   the masked region — first taken for normal engine variance, later shown by
   the [correctness audit](#10-correctness-audit-2026-09-14) to be four real
   bugs in the fork's kernels. After those fixes the engines agree to ≤4e-4
   on float tensors and ≤1 LSB on 8-bit output.

The evaluation phase was posted upstream to the maintainer in
[cool-japan/oxionnx#4](https://github.com/cool-japan/oxionnx/issues/4)
(comment `5653249873`); the engine fixes it describes landed in upstream
0.1.5–0.1.8. Everything after that is local engine work, kept in this
repository's vendored fork (see `../vendor/oxionnx/MODIFICATIONS.md`).

**Environment for every number below:** Windows 11, Intel i7-13620H
(6P+4E cores), Rust 1.94.0, OxiONNX 0.1.7 (vendored path dependency),
`lama_fp32.onnx` as shipped with the plug-in. Timing used interleaved A/B
rounds because the laptop's thermal state moves results by ±20%.

---

## 1. Does it run? Yes — and the fix is confirmed

We built a drop-in replacement worker (`lama-worker-oxionnx`, same CLI as the
ONNX-Runtime sidecar) and ran three reference images through both engines.

| Test | Image size | ORT time | OxiONNX 0.1.7 | Unmasked diff | Masked diff |
|---|---|---|---|---|---|
| Synthetic gradients | 512×512 | 8.8 s | 48.3 s | 0.0000 | 79.27 |
| Photo-like scene | 800×600 | 8.5 s | 47.3 s | 0.0000 | 129.63 |
| Odd size (mod-16 pad) | 1000×700 | 10.5 s | 79.1 s | 0.0000 | 106.20 |

**The unmasked region is bit-identical** (mean and max diff 0.0000) in all
three tests: outside the inpainting mask the two engines agree perfectly.
The masked region diverged — initially read as expected engine variance,
since LaMa *synthesises* that region. The [correctness audit](#10-correctness-audit-2026-09-14)
later proved that divergence was a set of real bugs (the masked output was
saturated white), not synthesis noise; with those fixed, the two engines
produce the same inpaint.

The model loads and executes end to end. The upstream `Slice` fix (negative
`starts[i]`/`ends[i]` via `saturating_add(dim)`) is confirmed working: the 98
`Pad` nodes that produced empty tensors in 0.1.4 now receive the correct
ONNX-canonical `[0,0,3,3,0,0,3,3]` pads from the
`_prepare_onnx_paddings` export chain.

## 2. Binary size: 5.5x smaller

| Worker | Size | Notes |
|---|---|---|
| ORT (`lama-worker.exe`) | 25,492,992 bytes | 24.3 MB, links `onnxruntime` + protobuf |
| OxiONNX (`lama-worker-oxionnx.exe`) | 4,630,016 bytes | 4.4 MB, zero C/C++ dependencies |

No protobuf, no MSVC ABI coupling, no DLL to ship alongside. For a plug-in
distributed to end users this is the most immediate practical win.

## 3. Initial inference speed: 5.5x slower than ORT

At first measurement OxiONNX was **~5.5x slower** than ORT's CPU path:

| Build | 512×512 | 1000×700 |
|---|---|---|
| OxiONNX (default features) | 48.3 s | 79.1 s |
| OxiONNX (+ `simd` feature) | 43.7 s | 70.0 s |
| ORT (CPU) | 8.8 s | 10.5 s |

The `simd` feature (AVX2 im2col + elementwise, off by default) helps by
10–15%. The `simd` and non-`simd` builds produce bit-identical output, so the
speedup carries no numerical cost. At this stage the reason for the gap
looked structural:

| Component | ORT | OxiONNX (as shipped) |
|---|---|---|
| GEMM | Hand-written assembly (AVX2/AVX-512/NEON), runtime dispatch | `matrixmultiply::sgemm` (pure Rust) |
| Conv2d | Direct + Winograd + im2col, all assembly-tuned | im2col + `matrixmultiply` |
| Elementwise | Hand-vectorised | AVX2 behind an opt-in feature |
| Parallelism | ORT thread pool | rayon around the im2col/sgemm calls |

Sections 5–7 below show that the gap was **not** in the kernels' quality but
in a handful of pathological code paths that profiling exposed, and that
after fixing them the ordering reverses.

## 4. CUDA: correct kernels, wrong workload

Tested with the `cuda` feature on an NVIDIA RTX 4050 Laptop GPU
(CUDA 13.2 driver):

| Build | 512×512 | Binary size |
|---|---|---|
| OxiONNX (CPU, `simd`) | 43.7 s | 4.4 MB |
| OxiONNX (CUDA) | 55.1 s | 6.0 MB |
| ORT (CPU) | 8.8 s | 24.3 MB |

The CUDA output is **bit-identical** to the CPU output (mean diff 0.0000), so
the OxiCUDA kernels are correct. It is nevertheless *slower* than the CPU
build for LaMa, for a structural reason: **62% of the graph's nodes are
CPU-only** (`Constant` 7,054, `Shape` 1,188, `Cast` 782, `Transpose` 638,
`Gather` 396, `Einsum` 216, …). Those shape and metadata ops force a
GPU→CPU→GPU round trip around every accelerated node.

CUDA can accelerate 6,636 nodes (38%) — the 222 `Conv`, 216 `MatMul`, 152
`Relu`, 75 `BatchNormalization`, and elementwise ops — but the interleaving
means the GPU spends most of its time waiting for uploads and readbacks. For
models with a higher compute-to-shuffle ratio (large transformer MatMul
blocks, few shape ops) the same dispatch layer would very likely show a real
speedup. LaMa's FFC architecture is an unusually shape-op-heavy graph.

*(Note: `Einsum` **is** accelerated by OxiCUDA, but only when the graph
placement decides it is worth it; the count above is what the graph exposes
to the provider layer.)*

## 5. Profiling: three nodes were 75% of the run

Per-node profiling (`OXIONNX_PROFILE=1`) attributed 27,967 ms of a 37,218 ms
run to **three `ConvTranspose` nodes** — the upsampling layers in LaMa's FFC
decoder. The implementation was a naive scatter-accumulate: for each input
pixel, scatter `in_val * w[k]` to `C_out × kH × kW` output positions, with no
vectorisation, no parallelism and no tiling.

| ConvTranspose | Before | After (direct output-oriented + rayon) | Speedup |
|---|---|---|---|
| model.24 (64→128, 512→256 ch) | 11,624 ms | 1,744 ms | 6.7x |
| model.27 (128→256, 256→128 ch) | 9,684 ms | 2,064 ms | 4.7x |
| model.30 (128→256, 128→64 ch) | 6,658 ms | 1,082 ms | 6.2x |
| **Total** | **27,967 ms** | **4,890 ms** | **5.7x** |

That single change took the full run from 48.3 s to 15.5 s, with the engine's
own output unchanged (all bit-identity claims in this section compare each
build against the previous one; the engine-vs-ORT audit is in §10).

## 6. The optimisation campaign

Eleven pathologies, found in this order by profiling, GEMM-shape
instrumentation, and microbenchmarks. Every row was verified with the
profiler and an image diff against the previous build — bit-identical by
construction. (This checked optimisation-preservation only; the §10 audit is
what caught the kernel bugs some of these rewrites had introduced.)

| # | Pathology | Fix | Impact |
|---|---|---|---|
| 1 | ConvTranspose scatter-accumulate (75% of runtime) | Direct output-oriented algorithm + rayon → AVX2 `vgatherdps` inline asm → row-wise contiguous-load kernel → 2-channel pairing | 27,967 → 205 ms |
| 2 | Pad: per-element `div`/`mod` coordinate maths | 4D NCHW reflect-pad fast path: memcpy interior, reflected edges | 1,638 → 112 ms |
| 3 | 98,446 per-batch GEMM calls of `m=64 k=64 n=1` | Merge both broadcast patterns into one dense GEMM (stack A along M; or `C' = B @ Aᵀ` when `N=1`) | 1,792 → 330 ms |
| 4 | `matrixmultiply` at 11 GFLOP/s for `M=3` (final conv `[3,64,7,7]`) | Direct convolution, no im2col, 24-wide AVX2 FMA, 9 accumulator chains | 909 → 66 ms |
| 5 | `parallel_sgemm` split M: every thread re-packed the whole B matrix (900 MB duplicate packing per conv) | Split N: B packed once | Conv total 2,894 → 1,410 ms |
| 6 | Generic transpose walked the FFC's rotations element-wise | `[outer][P][Q][inner]` block-swap decomposition (covers every perm in the graph) + AVX2 8×8 register tiles + `[P,2]↔[2,P]` interleave | 635 → 250 ms |
| 7 | `Slice`: per-element division odometer across 1,322 nodes | Bulk-copy fast path keyed on the last sliced axis | 325 → 207 ms |
| 8 | Every `Div` in the graph is `x / scalar`, taking a 5–7 cycle/element scalar broadcast walk | IEEE-exact broadcast `vdivps` path | 251 → 153 ms |
| 9 | Single-batch Einsum contractions ran single-threaded (72 GFLOP/s) | N-split parallel GEMM | 409 → 342 ms |
| 10 | BatchNorm: scalar loop plus a redundant full copy pass | Single-pass SIMD channel kernel | 196 → 107 ms |
| 11 | 198 MB protobuf parse on every run | `save_optimized`/`load_optimized` session cache, model size+mtime checked, `SESSION_CACHE_REVISION` keyed, written atomically | ~500 ms → ~150 ms per run |

Two further graph-level folds were later adopted from ONNX Runtime's
optimizer design (see `ORT_STRATEGIES.md`):

| # | Pathology | Fix | Impact |
|---|---|---|---|
| 12 | 72 × `BatchNorm(Add(conv_a, conv_b))` — the FFC block shape — which ORT reaches via `conv_add_fusion` + `conv_bn_fusion` | `fuse_add_batchnorm`: normalisation distributes over the sum and folds into **both** branches' weights | 72 nodes removed; −53 ms A/B |
| 13 | 3 × `ConvTranspose → BatchNorm` in the decoder (ORT does not fold this shape) | `fuse_conv_batchnorm` extended to `ConvTranspose` (channel dim is weight dim 1) | ConvTranspose 205 → 167 ms |

### The final layout & slot pass (2026-09-15)

Three more changes, discovered by reading the *other half* of the profile —
the ops whose per-node time is pure data movement:

| # | Pathology | Fix | Impact (node-time, 512×512) |
|---|---|---|---|
| 14 | `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten` never used the engine's in-place path: every node allocated and copied a payload it did not change (and `Reshape`'s slot path did it **twice**) | `supports_inplace`/`execute_inplace` for the four metadata-only ops; `Reshape`'s slot fallback collapsed to one copy | Reshape 303 → 86 ms; Squeeze 21 → 0; Unsqueeze 45 → 0 |
| 15 | `Concat`/`Slice`/`Transpose` claimed output slots but used the trait's default `execute_into_slots` (materialise a result, then copy it into the slot); `Pad`'s override did the same | Direct-to-slot helpers `concat_into`, `slice_into`, `transpose_into`, `pad_axes_into` | one full copy per node removed (mostly L2-resident, so a modest wall gain) |
| 16 | Every slot write was preceded by a full **zeroing pass** over the recycled buffer (`SizeClassPool::acquire`) that the kernel then overwrote | `fully_writes_slots()` predicate on `Operator` (default `false`) + `acquire_for_overwrite` for the audited ops (Conv, ConvTranspose, MatMul, the shape ops, Pad, Gather, Cast, Relu, Sigmoid) | ~0.2 s/run at 512×512 |

Verified: engine suite 955 tests across 60 binaries; the three reference
images are **bit-identical** to the previous build (pure data movement, no
arithmetic touched); interleaved A/B on 512×512 and 800×600 shows **−4.2%
wall** on both. The ORT ratio moves from 1.92x to **1.97x**.

### The arc

| Stage | 512×512 total | What landed |
|---|---|---|
| OxiONNX 0.1.7, no tuning | 48.3 s | — |
| + ConvTranspose algorithm rewrite | 15.5 s | item 1 (first form) |
| + AVX2 kernels, Pad fast path | 11.9 s | items 1–2 |
| + N-split GEMM, merged MatMul, direct small-M | 9.0 s | items 3–5 |
| + Slice / Div / Transpose / Einsum / 2-ch ConvTranspose / BatchNorm SIMD | 4.5 s | items 6–10 |
| + session cache, ORT-inspired fusions | **3.93 s** | items 11–13 |

## 7. Final result

Six interleaved A/B rounds, same minutes, to control for thermal state:

| Round | ORT | OxiONNX |
|---|---|---|
| 0 | 7.54 s | **3.94 s** |
| 1 | 7.45 s | **3.93 s** |
| 2 | 7.61 s | **3.95 s** |
| 3 | 7.62 s | **3.96 s** |
| 4 | 7.63 s | **3.93 s** |
| 5 | 7.55 s | **3.96 s** |
| **Average** | **7.57 s** | **3.95 s** |

**OxiONNX is 1.92x faster than ORT** on this workload, wins every
interleaved round, and ships in a 4.7 MB binary versus 24.3 MB. (The output
comparison in those rounds only checked the unmasked region and surfaces
that the composite replaces outside the mask; the masked region was broken
by the bugs the [correctness audit](#10-correctness-audit-2026-09-14) later
found and fixed. With the fixed engine the same A/B still holds - the fixes
cost no measurable time.)

A later re-measurement after the final layout/slot pass (see the end of §6)
put the same interleaved comparison at **ORT 7.37–7.57 s vs OxiONNX
3.73–3.77 s (load + run) — 1.97x**.

The remaining node-execution profile (3,527 ms total) is headed by
`Conv` (~1,280 ms, of which the 3×3 im2col gathers and GEMMs are the bulk),
`Einsum` (355 ms), `Transpose` (250 ms) and `Slice` (212 ms).

## 8. What didn't work

Both were measured, found slower, and reverted; they are recorded so they
are not retried blind.

| Attempt | Rationale | Measurement |
|---|---|---|
| Custom small-K AVX2 GEMM for the Einsum contractions (`m=64, k=64`) | `matrixmultiply`'s per-call packing never amortises over a 64-wide K loop | **Slower**: 1,101 ms vs 352 ms for the op total. The broadcast-per-FMA structure is port-5-bound (~4 MACs/instruction) |
| Row-block parallel ConvTranspose (task = rows × all channels) | Keep each task's input slice cache-resident | **Slower**: 387 ms vs 354 ms. On this CPU the 8.4 MB input already stays in L3 across the fine-grained per-plane tasks; coarse tasks only cost load balance |

Earlier, before the row-wise kernel existed, an AVX2 intrinsics gather kernel
and an AVX2 **inline-assembly** `vgatherdps` kernel were both tried: the
inline asm version won at the time (4,890 → 800 ms), but was later superseded
and deleted by the row-wise kernel, which removed the gathers altogether and
was 2.4x faster still. Inline asm only wins when the loop is
instruction-bound; that one was gather-port-bound.

## 9. What is left (v2 candidates)

> **Status (2026-09-15):** the Reshape/Concat half of this list was addressed
> in the final layout/slot pass (§6) — the reshape family is now zero-copy
> on the common path and the sequence ops single-pass. The Conv im2col and
> `session.run` scaffolding items remain open.

The remaining ~160 ms to reach exactly 2.0x is structural:

- **Conv im2col (~300 ms theoretical).** Every 3×3 conv still materialises a
  19–56 MB column matrix per node. ONNX Runtime's segmented working buffer
  (64 KB per thread, L2-resident) or a full implicit-GEMM kernel would remove
  that traffic.
- **Reshape/Concat memcpy floor (~420 ms combined).** These ops copy
  multi-megabyte intermediates; a stride-based `Tensor` (Arc + offset) would
  make reshape zero-copy.
- **`session.run` scaffolding (~270 ms).** ≈27 µs per node of input
  gathering, shape resolution and output insertion across 9,900 nodes.

## Postscript: upstream

The 0.1.4 failures (empty-pads `Pad` panic from a `Slice` clamp bug, and the
`Einsum` rank error that followed it) were reported with a full reproduction
and a local patch in
[cool-japan/oxionnx#4](https://github.com/cool-japan/oxionnx/issues/4). The
maintainer identified the real root cause (negative `Slice` steps being
clamped, not `Pad` leniency), fixed it in 0.1.5, and additionally hardened the
legacy opset ≤ 10 `Pad` attribute form for 0.1.8. The length-normalisation
part of our patch was correctly declined as symptom-treatment. The vendored
fork here carries only performance work; the correctness fixes are upstream's.

---

## 10. Correctness audit (2026-09-14)

After the GIMP bridge was installed and exercised, the inpainting result in
GIMP came back as a **white rectangle** inside the selection. That report
unwound into two layers of problems.

### 10.1 The white output: a worker port bug

The LaMa model outputs **0–255** floats; both the reference Python worker and
the ORT worker divide by 255 before compositing. The OxiONNX worker — ported
from that contract — **skipped the division**, so every model output value
above 1.0 clamped to white in the masked region. Outside the mask the
soft-mask composite restores the original pixels, which is why every unmasked
comparison (and every earlier "bit-identical" check between OxiONNX builds)
looked perfect while the masked content was broken. Fixed in the worker
(`out_hwc = raw * 1/255`).

Once the scaling was fixed, the masked content appeared — but it did **not**
match ORT.

### 10.2 The remaining divergence: four engine bugs

Method: build a **tap-augmented model** (every N-th node output promoted to a
graph output), run ORT (optimisations disabled) and OxiONNX
(`OXIONNX_OPT_LEVEL=none`) on identical inputs, and diff tensor by tensor.
The first divergent node localises the bug; the diff structure (which
rows/columns/positions) then identifies it. Then hand-compute the expected
value from the model weights to confirm.

The audit found four bugs, all introduced by the fork's optimisation work
(none present upstream — and all invisible to the earlier self-referential
comparisons):

| # | Where | Bug | Symptom |
|---|---|---|---|
| 1 | `oxionnx-ops/src/conv/transpose.rs` (both AVX2 s2k3 kernels) | On odd output rows the **ky=0 and ky=2 weight sets were paired with the wrong input rows** (the two row vectors were swapped) | All odd output rows of every ConvTranspose wrong by ~5–10% |
| 2 | same file | On the **last output row** the ky=0 input row is `(oy+1)/2 == h` and does not exist; the SIMD path read **out of bounds** and used the garbage | Last row wildly wrong (up to ~2000) |
| 3 | `oxionnx-ops/src/conv/conv2d.rs` (`conv2d_direct_small_m_f32`) | The direct small-M kernel indexed weights as **`[c_in][c_out][k][k]`** instead of ONNX `[c_out][c_in][k][k]` | The final 7×7 conv (→ sigmoid → mask multiply) wrong everywhere; the whole masked region off by ~10% |
| 4 | `transpose.rs` (`try_convtranspose_as_conv` fallback) | Passed the **original stride** to the inner convolution (the zero-interleaved input already encodes it → output size shrank and the copy-back panicked), and never **flipped the kernel** spatially | Any non-s2k3 ConvTranspose crashed or mis-computed (pre-existing; surfaced by the engine test suite) |

Fixes: correct the u/v weight pairing; route the bottom row through the
bounds-checked scalar tail; fix the small-M weight layout (both SIMD loops
and the scalar tail); use stride 1 + spatial kernel flip in the fallback.

### 10.3 Verification after the fixes

- **Tap-level**: all sampled intermediate tensors agree with ORT to ≤4e-4
  (max over 17478-node graph; most taps ≤1e-6). Final model output max diff
  4e-4 out of 0–255.
- **End-to-end 8-bit**: the 512×512 fixture pair is **bit-identical** to ORT
  inside and outside the mask; the 800×600 GIMP-exported input differs by at
  most **1 LSB** inside the mask (mean 0.00), 0 outside.
- **Engine test suite**: 955 tests across 60 binaries pass, including new
  regression tests that compare the s2k3 fast path, the fallback, and the
  small-M conv against naive reference implementations with asymmetric
  weights and odd-dimension cases.
- **Performance**: unchanged — the fixes touch indexing and one scalar tail
  row per ConvTranspose plane; interleaved measurement still shows ~4.2 s vs
  ORT's ~10 s on the same session.

### 10.4 Lessons learned

These are the durable lessons from the mistake, recorded so the same class of
failure is caught much earlier next time.

**1. Self-consistent is not correct — never validate an engine against
itself.** The whole optimisation campaign compared each build against the
previous OxiONNX build ("output unchanged"). That test can only prove an
optimisation *preserved* behaviour; it can never prove the behaviour was
right. A wrong engine passes it forever. Every kernel change needs at least
one comparison against an **independent** implementation (ORT here).

**2. Choose comparison regions that can actually fail.** The soft-mask
composite replaces everything outside the mask with the original pixels, so
the unmasked region is *always* bit-identical no matter how broken the model
output is. And the missing `÷255` saturated the masked region to white, which
swamped every other error. The only region with real signal was the masked
region — and that was the one region the original A/B numbers never checked
visually. Rule: when validating an inpainting pipeline, inspect the
synthesised content explicitly, and compare the **raw model output** (before
compositing) whenever possible.

**3. Port contracts must be verified line by line.** The `÷255` existed in
both reference workers (Python and ORT); the port dropped it silently and
nothing in the fork ever looked at model-output magnitudes. When porting an
op or a worker, diff the data-flow contract against the reference
implementation: input scaling, output scaling, masks, alpha, colour order,
layout.

**4. Optimised kernels break at parity and boundary cases, not in the
middle.** The three kernel bugs were: odd output *parity rows* (ConvTranspose
weight pairing), the *last* row (out-of-bounds ky=0 tap), and a *layout*
assumption (small-M weight indexing `[ci][co]` vs ONNX `[co][ci]`). Symmetric
or interior-only test data passes all three. Regression tests must use
asymmetric weights, odd dimensions, and explicitly include first/last
rows/columns.

**5. Differential testing localises the bug in minutes.** The tap harness
(every N-th node output promoted to a graph output, both engines run with
optimisations off, tensors diffed in evaluation order) reduced a
17,480-node graph to a single first-divergent node in two bisection rounds;
hand-computing the expected values from the weights then identified the
exact index error. This, not guesswork, is the tool for any future "engine X
disagrees with engine Y" problem — the reusable implementation lives in
[`tools/tap_diff.py`](../tools/tap_diff.py) with usage in
[`tools/README.md`](../tools/README.md).

**6. Know your tolerance budget — and what exceeds it.** Accumulation-order
differences across a 17k-node network give ~1e-6 per tap and ≤1 LSB at
8-bit; anything at 1e-2 or above is a bug, not noise. The audit's first
divergent node differed by 3158, four orders of magnitude above noise — an
obvious bug once measured, invisible while unmeasured.

**7. Put the debug artifacts where the failure happens.** The exchanged
image/mask/result PNGs are deleted with the temp directory after every
run — which is exactly when you need them. The bridge now keeps copies with
`LAMA_OXIONNX_DEBUG_DIR` set, and the whole GIMP path is exercisable
headlessly in batch mode (see [`gimp/README.md`](../gimp/README.md)). The
white rectangle was reproduced and diagnosed in one such run.

**8. A cheap invariant would have caught the white bug on day one.** A
worker-side sanity check — "if every masked pixel is saturated, fail or
warn" — is three lines and would have surfaced the missing `÷255`
immediately, before any performance work. Cheap invariants beat expensive
archaeology.

**9. State coverage when claiming verification.** "Verified" without saying
*what was covered* is how a broken build reads as done for a whole session.
The claim "all three reference images bit-identical" was true and vacuous at
the same time: bit-identical **to the previous OxiONNX build**, in the one
region the composite guaranteed anyway.
