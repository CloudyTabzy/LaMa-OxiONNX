// Modified by the GIMP LaMa inpainting fork of OxiONNX (2026-09): ADDS a direct small-M conv kernel, an N-split parallel GEMM and a reusable im2col scratch buffer.
// See ../../../MODIFICATIONS.md for the full change list and rationale.

//! 2D convolution: the specialised, vectorised and parallelised spatial-rank-2
//! kernel.
//!
//! Ranks other than 2 are handled by [`super::conv_nd`], which lowers 1D to
//! this kernel (`[N, C, W]` → `[N, C, 1, W]`) and runs a generic im2col + GEMM
//! for rank ≥ 3.

use oxionnx_core::{OnnxError, Tensor};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::im2col::im2col_adaptive;
use super::spatial;
use super::winograd::conv2d_winograd_f2x3;

/// Compute the output shape for a 2D convolution.
///
/// `input_shape` must be `[N, C, H, W]`.
/// `weight_shape` must be `[F, C/group, kH, kW]`.
/// `pads` must be `[pad_top, pad_left, pad_bottom, pad_right]` (length 4).
///
/// Returns a typed [`OnnxError::ShapeMismatch`] — never panics — for any
/// model-derived combination that cannot produce a valid output extent
/// (rank mismatch, zero stride/dilation, padded input below the dilated
/// kernel extent, or a size computation that overflows).
pub(crate) fn compute_conv2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilations: &[usize],
) -> Result<Vec<usize>, OnnxError> {
    if input_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "Conv: input must be 4D [N,C,H,W], got rank {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "Conv: weight must be 4D [F,C/group,kH,kW], got rank {}",
            weight_shape.len()
        )));
    }
    let strides2 = [
        strides.first().copied().unwrap_or(1),
        strides.get(1).copied().unwrap_or(1),
    ];
    let dilations2 = [
        dilations.first().copied().unwrap_or(1),
        dilations.get(1).copied().unwrap_or(1),
    ];
    let pads4 = [
        pads.first().copied().unwrap_or(0),
        pads.get(1).copied().unwrap_or(0),
        pads.get(2).copied().unwrap_or(0),
        pads.get(3).copied().unwrap_or(0),
    ];
    spatial::compute_conv_out_shape(
        "Conv",
        input_shape,
        weight_shape,
        &strides2,
        &pads4,
        &dilations2,
    )
}

/// Whether the Winograd F(2,3) path is expected to beat im2col + SGEMM for a
/// layer of this size.
///
/// Measured on this implementation (`perf_probe_winograd_vs_im2col`, Apple M-series,
/// release build, single thread): Winograd is **slower** than im2col + SGEMM at
/// every realistic CNN layer size, by 1.34× (8 channels, 64×64) up to 2.81×
/// (3→64 channels, 224×224). The cause is structural rather than the filter
/// transform: the accumulation re-streams the whole `oc * c * 16` transformed
/// filter bank once per 2×2 output tile, so its traffic grows as
/// `oc * c * tiles` while GEMM's grows as `oc * c + c * tiles`.
///
/// The path is therefore retained only below `WINOGRAD_MAX_WORK`, where the
/// absolute cost of either algorithm is a few microseconds and the choice is
/// immaterial — keeping the long-standing reference values of the small
/// Winograd fixtures bit-identical — and declined above it, where the measured
/// 1.3–2.8× SGEMM advantage applies. Because the two algorithms differ in
/// rounding (≈1e-5 relative), the threshold sits an order of magnitude above
/// every shape exercised by the test-suite so no existing expectation moves.
const WINOGRAD_MAX_WORK: usize = 4096;

/// Cost proxy for the Winograd path: transformed-filter traffic per image.
fn winograd_work(oc: usize, c: usize, oh: usize, ow: usize) -> usize {
    oc.saturating_mul(c)
        .saturating_mul(oh.div_ceil(2))
        .saturating_mul(ow.div_ceil(2))
}

/// Upper bound on the bytes a **single** im2col column workspace may occupy.
///
/// The column matrix is `[C/group * kH * kW, OH * OW]` f32, which grows as the
/// product of the kernel volume and the output area: `inswapper_128`'s last
/// convolution (`[3, 128, 7, 7]` over a `128×128` output) alone asks for
/// 411 MB, and `[256, 512, 3, 3]` over `128×128` for 302 MB — every frame,
/// on top of the resident weights. On `wasm32` a `WebAssembly.Memory.grow`
/// that cannot be satisfied *aborts* the module, and wasm linear memory never
/// shrinks, so one such request permanently raises the tab's memory ceiling.
///
/// Above this cap the output columns are processed in blocks (see
/// [`im2col_block_cols`]): the workspace is rebuilt per block and the GEMM
/// writes each block into the disjoint columns of the output it owns. The
/// result is bit-identical — `matrixmultiply`'s N loop is its outermost loop,
/// so the K-accumulation order of every output element is unchanged by an
/// external split of N (asserted by
/// `sgemm_column_blocks_are_bitwise_identical_to_one_call`).
///
/// The cap is **per workspace**, not per call: the native multi-job path keeps
/// one capped buffer per rayon worker (`for_each_init`), so its aggregate peak
/// is `workers × cap` — still a strict improvement on `workers × full`. The
/// wasm32 twin and the single-job path hold exactly one.
pub(crate) const IM2COL_WORKSPACE_MAX_BYTES: usize = 64 * 1024 * 1024;

// Thread-local reusable im2col scratch buffer.
//
// The single-job path allocates its column matrix once per convolution node.
// For the FFC workhorse shapes (`c_in=384, k=3`, output `64x64`) that is a
// ~56 MB `vec![0.0; N]` allocated, zero-filled and page-faulted in for every
// one of the ~108 nodes. Reusing one allocation across nodes removes that
// fixed cost after the first use.
//
// Sound because `im2col_adaptive` (all three dispatch targets) and
// `im2col_block` unconditionally overwrite every element they read, so a
// dirty scratch buffer is never observed. `take`/`return` both happen on the
// calling executor thread, so a thread-local is the whole synchronisation —
// the rayon workers only ever borrow the slice passed to them.
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static COL_SCRATCH: core::cell::RefCell<Vec<f32>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

/// Is the im2col scratch cache enabled? Read once — this sits on the
/// per-convolution path, and `env::var_os` walks the process environment on
/// every call. `OXIONNX_NO_SCRATCH_CACHE=1` restores fresh-allocation
/// behaviour for A/B bisecting.
#[cfg(not(target_arch = "wasm32"))]
fn scratch_cache_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("OXIONNX_NO_SCRATCH_CACHE").is_none())
}

/// Number of output columns one im2col + GEMM block covers.
///
/// Returns `col_cols` — i.e. *no* blocking, the historical code path, byte for
/// byte — whenever the full workspace already fits `cap_bytes`, so small and
/// medium convolutions keep their exact previous behaviour and cost.
///
/// Over the cap it picks the smallest number of blocks that fits,
/// `ceil(col_cols / cap_cols)`, then spreads the columns evenly over them
/// rather than filling greedily (a greedy split leaves a lopsided remainder
/// block, e.g. 14563 + 1821). The even width is finally rounded to a whole
/// number of output rows (`ow`) when that still fits, so each block is a
/// contiguous run of output rows and the gather never has to handle a partial
/// row — true for every convolution in `inswapper_128`. Only a single output
/// row wider than the cap falls back to an unaligned width.
pub(crate) fn im2col_block_cols(
    col_rows: usize,
    col_cols: usize,
    ow: usize,
    cap_bytes: usize,
) -> usize {
    let bytes_per_col = col_rows.saturating_mul(core::mem::size_of::<f32>());
    if bytes_per_col == 0 || col_cols == 0 {
        return col_cols;
    }
    if bytes_per_col.saturating_mul(col_cols) <= cap_bytes {
        return col_cols;
    }
    // At least one column always fits, however large a single column is.
    let cap_cols = (cap_bytes / bytes_per_col).max(1);
    let blocks = col_cols.div_ceil(cap_cols);
    let even = col_cols.div_ceil(blocks.max(1)).clamp(1, cap_cols);
    if ow > 0 && even >= ow {
        // Round up to whole output rows if that still fits, else down.
        let up = even.next_multiple_of(ow);
        if up <= cap_cols {
            return up.min(col_cols);
        }
        let down = (even / ow) * ow;
        if down >= 1 {
            return down.min(col_cols);
        }
    }
    even
}

/// Write conv2d result directly into a pre-allocated output buffer.
///
/// `out_shape` must be the result of `compute_conv2d_out_shape` for these inputs.
/// `out` must have length equal to `out_shape.iter().product()`.
///
/// Degenerate parameters that would divide by zero or index out of range
/// (`group == 0`, a non-4D input/weight, a zero-volume output) leave `out`
/// zero-filled instead of panicking; callers on the model-execution path
/// reject them earlier with a typed error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) {
    conv2d_into_slices(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        strides,
        pads,
        dilations,
        group,
        out,
        out_shape,
    );
}

/// Slice-based core of [`conv2d_into`].
///
/// Taking raw slices plus shapes (rather than [`Tensor`]s) lets the rank-1
/// lowering in [`super::conv_nd`] re-interpret a `[N, C, W]` buffer as
/// `[N, C, 1, W]` with no copy, and lets the typed (f16/bf16) wrappers pass
/// their promoted scratch buffers directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_into_slices(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) {
    conv2d_into_slices_capped(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        strides,
        pads,
        dilations,
        group,
        out,
        out_shape,
        IM2COL_WORKSPACE_MAX_BYTES,
    );
}

/// [`conv2d_into_slices`] with the im2col workspace cap as a parameter.
///
/// Production callers go through [`conv2d_into_slices`], which passes
/// [`IM2COL_WORKSPACE_MAX_BYTES`]. The explicit cap exists so the test-suite
/// can reach *both* sides of the blocking decision on the same shape —
/// `usize::MAX` forces the monolithic reference, a small cap forces many
/// blocks — and assert they agree bit for bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_into_slices_capped(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
    workspace_cap_bytes: usize,
) {
    // Defensive guards: never panic on malformed shapes/attributes.
    if group == 0 || input_shape.len() != 4 || weight_shape.len() != 4 || out_shape.len() != 4 {
        out.fill(0.0_f32);
        return;
    }
    let n = input_shape[0];
    let c_in = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];
    let c_out = weight_shape[0];
    let c_per_group = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let oh = out_shape[2];
    let ow = out_shape[3];

    // Every branch below fully overwrites the first `needed` elements, so the
    // buffer is not pre-zeroed (that memset used to cost a full pass over the
    // largest tensors in a CNN). Only a caller-supplied tail beyond `needed`
    // — which no in-tree caller passes — is cleared, preserving the previous
    // contract exactly.
    let needed: usize = out_shape.iter().product();
    if out.len() < needed || out.is_empty() || oh == 0 || ow == 0 {
        out.fill(0.0_f32);
        return;
    }
    if out.len() > needed {
        out[needed..].fill(0.0_f32);
    }
    // Reject buffers too small for the declared geometry rather than indexing
    // past their end inside the GEMM.
    if input.len() < n * c_in * h * w
        || weight.len() < c_out * c_per_group * kh * kw
        || c_out % group != 0
        || c_per_group * group != c_in
    {
        out.fill(0.0_f32);
        return;
    }

    let c_out_per_group = c_out / group;
    let no_pad = pads == [0, 0, 0, 0];

    // ── 1×1 conv fast path: skip im2col entirely ──────────────────────────
    // When kernel is 1×1, stride=1, no padding, dilation=1: input channels
    // are already contiguous per spatial position → direct matmul.
    if kh == 1 && kw == 1 && strides == [1, 1] && no_pad && dilations == [1, 1] {
        conv2d_1x1_into(
            input,
            weight,
            bias,
            n,
            c_in,
            c_out,
            c_per_group,
            c_out_per_group,
            h,
            w,
            group,
            out,
        );
        return;
    }

    // ── Direct small-M fast path ──────────────────────────────────────────
    // For very few output channels (M ≤ 3) the im2col + GEMM path is
    // pathological: `matrixmultiply` measures ~11 GFLOP/s at M=3 (vs ~100
    // at M≥64) because its microkernel wastes most of its tile. The final
    // LaMa conv (weight [3, 64, 7, 7], M=3) is exactly this shape and
    // dominated the whole Conv profile at ~900 ms.
    //
    // This path skips im2col entirely and computes each output row
    // directly, keeping the accumulator in ymm registers with the output
    // channel loop unrolled. It reads the input once (cache-resident
    // re-reads for the kernel taps) and streams B exactly once.
    #[cfg(target_arch = "x86_64")]
    if c_out <= 3
        && group == 1
        && strides == [1, 1]
        && dilations == [1, 1]
        && c_in >= 4
        && oh * ow >= 4096
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        // Materialize the padded input when needed (rare in LaMa: the
        // padding is usually an explicit Pad node upstream).
        if no_pad {
            // SAFETY: AVX2+FMA confirmed above; shapes validated by the
            // caller; `out` has `needed` writable elements.
            unsafe {
                conv2d_direct_small_m_f32(
                    input, n, c_in, h, w, weight, bias, c_out, kh, kw, oh, ow, out,
                );
            }
        } else {
            let (padded, ph, pw) =
                pad_input_for_direct(input, n, c_in, h, w, pads);
            unsafe {
                conv2d_direct_small_m_f32(
                    &padded, n, c_in, ph, pw, weight, bias, c_out, kh, kw, oh, ow, out,
                );
            }
        }
        return;
    }

    // ── Winograd F(2,3) fast path: 3×3 kernel, stride 1, dilation 1 ──────
    if kh == 3
        && kw == 3
        && strides == [1, 1]
        && dilations == [1, 1]
        && group == 1
        && oh >= 4
        && ow >= 4
        && pads[0] == pads[1]
        && pads[0] == pads[2]
        && pads[0] == pads[3]
        && winograd_work(c_out, c_in, oh, ow) <= WINOGRAD_MAX_WORK
    {
        let pad = pads[0];
        if let Ok(data) = conv2d_winograd_f2x3(input, weight, bias, n, c_in, h, w, c_out, pad) {
            // Length is guaranteed by construction, but this file must stay
            // panic-free even if a future edit changes the extent formula.
            if data.len() == needed {
                out[..needed].copy_from_slice(&data);
                return;
            }
        }
    }

    let col_rows = c_per_group * kh * kw;
    let col_cols = oh * ow;
    let total_jobs = n * group;
    // `col_cols` (no blocking) for every shape whose workspace fits the cap.
    let block_cols = im2col_block_cols(col_rows, col_cols, ow, workspace_cap_bytes);

    if total_jobs <= 1 {
        // ── Single job: im2col + GEMM straight into the output buffer ────
        //
        // This is the standard batch-1, group-1 inference shape. Both halves
        // are split across rayon when the work justifies it: im2col by input
        // channel (disjoint row bands of the column matrix) and the GEMM by
        // output channel (disjoint row bands of the result), so the arithmetic
        // per element — and therefore the result — is bit-identical to the
        // sequential path.
        //
        // Over the workspace cap the output columns are additionally walked in
        // blocks of `block_cols`: the same `col` buffer is refilled for each
        // block and the GEMM writes it into the columns of `out` that block
        // owns (row stride `col_cols`, hence the `_ldc` entry point). Both
        // halves stay bit-identical — im2col is a pure gather, and the GEMM's
        // K-accumulation order does not depend on how N is split.
        #[cfg(not(target_arch = "wasm32"))]
        let use_scratch_cache = scratch_cache_enabled();
        #[cfg(not(target_arch = "wasm32"))]
        if !use_scratch_cache {
            let mut col = vec![0.0f32; col_rows * block_cols];
            single_job_parallel(
                input,
                weight,
                bias,
                strides,
                pads,
                dilations,
                c_in,
                h,
                w,
                c_per_group,
                c_out_per_group,
                kh,
                kw,
                oh,
                ow,
                n,
                group,
                c_out,
                col_rows,
                col_cols,
                block_cols,
                &mut col,
                out,
            );
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        COL_SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            let scratch_len = col_rows * block_cols;
            if buf.len() < scratch_len {
                buf.resize(scratch_len, 0.0_f32);
            }
            single_job_parallel(
                input,
                weight,
                bias,
                strides,
                pads,
                dilations,
                c_in,
                h,
                w,
                c_per_group,
                c_out_per_group,
                kh,
                kw,
                oh,
                ow,
                n,
                group,
                c_out,
                col_rows,
                col_cols,
                block_cols,
                &mut buf[..scratch_len],
                out,
            );
        });
        #[cfg(target_arch = "wasm32")]
        {
            let mut col = vec![0.0f32; col_rows * block_cols];
            single_job_parallel(
                input,
                weight,
                bias,
                strides,
                pads,
                dilations,
                c_in,
                h,
                w,
                c_per_group,
                c_out_per_group,
                kh,
                kw,
                oh,
                ow,
                n,
                group,
                c_out,
                col_rows,
                col_cols,
                block_cols,
                &mut col,
                out,
            );
        }
    } else {
        // ── Parallel path: multiple (batch, group) jobs ─────────────────
        //
        // Each job writes straight into its slice of `out` — no per-job
        // `Vec<f32>` collected and then `copy_from_slice`d back — mirroring
        // the batched-matmul fix in `math_typed::matmul_f32_into`. Job `idx`
        // (`= batch * group + g`) owns `out[idx*job_out_size..][..job_out_size]`:
        // since `c_out == group * c_out_per_group` is already guaranteed by
        // the `c_out % group != 0` guard above, `(batch * c_out + g *
        // c_out_per_group) * col_cols` — the offset the sequential branch
        // above uses — is exactly `idx * job_out_size`, so this is the same
        // destination, just written directly instead of staged through an
        // intermediate buffer.
        //
        // The im2col scratch buffer is amortised across every job one rayon
        // worker handles via `for_each_init` (same pattern as
        // `attention::core::SdpaJob::run_parallel`), rather than allocated
        // fresh per job as before. Reusing a *dirty* scratch buffer across
        // jobs is sound because `im2col_adaptive` (all three of its dispatch
        // targets) unconditionally overwrites every element of its output —
        // including explicit zero-fills for padding — so nothing is ever
        // read from `col` before this job writes it.
        let job_out_size = c_out_per_group * col_cols;
        // `c_out_per_group == 0` (a weight with zero output channels) makes
        // `job_out_size == 0`; `par_chunks_mut(0)` panics, and there is
        // nothing to write in that case anyway (the top-of-function
        // `out[needed..].fill(0.0)` already zeroed everything, since
        // `needed == total_jobs * job_out_size == 0` too).
        if job_out_size > 0 {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            out[..total_jobs * job_out_size]
                .par_chunks_mut(job_out_size)
                .enumerate()
                .for_each_init(
                    || vec![0.0f32; col_rows * block_cols],
                    |col_scratch, (idx, dst)| {
                        conv2d_single_job_into(
                            input,
                            weight,
                            bias,
                            strides,
                            pads,
                            dilations,
                            c_in,
                            h,
                            w,
                            c_per_group,
                            c_out_per_group,
                            kh,
                            kw,
                            oh,
                            ow,
                            col_rows,
                            col_cols,
                            block_cols,
                            idx / group,
                            idx % group,
                            col_scratch,
                            dst,
                        );
                    },
                );

            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut col_scratch = vec![0.0f32; col_rows * block_cols];
                for (idx, dst) in out[..total_jobs * job_out_size]
                    .chunks_mut(job_out_size)
                    .enumerate()
                {
                    conv2d_single_job_into(
                        input,
                        weight,
                        bias,
                        strides,
                        pads,
                        dilations,
                        c_in,
                        h,
                        w,
                        c_per_group,
                        c_out_per_group,
                        kh,
                        kw,
                        oh,
                        ow,
                        col_rows,
                        col_cols,
                        block_cols,
                        idx / group,
                        idx % group,
                        &mut col_scratch,
                        dst,
                    );
                }
            }
        }
    }
}

/// 2D convolution via im2col + GEMM with Rayon parallelization.
///
/// Each (batch, group) pair is independent and processed in parallel on
/// native targets.  On WASM (single-threaded), falls back to sequential.
///
/// input: [N, C_in, H, W]
/// weight: \[C_out, C_in/group, kH, kW\]
/// bias: \[C_out\] (optional)
///
/// Parameter combinations with no valid output extent (padded input below the
/// dilated kernel, zero stride/dilation, non-4D operands) yield an empty
/// `[0, 0, 0, 0]` tensor rather than a panic. The `Conv` operator on the model
/// path uses `compute_conv2d_out_shape` directly and surfaces the typed error.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
) -> Tensor {
    let Ok(out_shape) =
        compute_conv2d_out_shape(&input.shape, &weight.shape, &strides, &pads, &dilations)
    else {
        return Tensor::new(Vec::new(), vec![0, 0, 0, 0]);
    };
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv2d_into(
        input, weight, bias, strides, pads, dilations, group, &mut data, &out_shape,
    );
    Tensor::new(data, out_shape)
}

/// Minimum `M * K * N` before a GEMM (or an im2col of comparable size) is worth
/// splitting across threads. Shared by the 1×1 path and the single-job path so
/// small convolutions keep their sequential, allocation-free behaviour.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const PARALLEL_GEMM_THRESHOLD: usize = 64 * 64 * 64;

/// Run `im2col_adaptive`, splitting the column matrix by input channel across
/// rayon when the work justifies it.
///
/// Row `r` of the column matrix belongs to input channel `r / (kH * kW)`, so a
/// contiguous band of input channels owns a contiguous band of rows: each
/// thread calls the *same* `im2col_adaptive` with a shifted `in_c_start` and a
/// disjoint sub-slice of `col`. The gather is element-wise, so the result is
/// byte-identical to the sequential fill.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_maybe_parallel(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    oh: usize,
    ow: usize,
    batch: usize,
    col: &mut [f32],
) {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        let col_cols = oh * ow;
        let num_threads = rayon::current_num_threads();
        let work = c_per_group * kh * kw * col_cols;
        if num_threads > 1 && c_per_group >= 2 && work >= PARALLEL_GEMM_THRESHOLD {
            let chunk_ic = c_per_group.div_ceil(num_threads).max(1);
            let chunk_len = chunk_ic * kh * kw * col_cols;
            col.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(t, sub)| {
                    let first = t * chunk_ic;
                    if first >= c_per_group {
                        return;
                    }
                    let count = (c_per_group - first).min(chunk_ic);
                    im2col_adaptive(
                        input,
                        c_in,
                        h,
                        w,
                        in_c_start + first,
                        count,
                        kh,
                        kw,
                        strides,
                        pads,
                        dilations,
                        oh,
                        ow,
                        batch,
                        sub,
                    );
                });
            return;
        }
    }

    im2col_adaptive(
        input,
        c_in,
        h,
        w,
        in_c_start,
        c_per_group,
        kh,
        kw,
        strides,
        pads,
        dilations,
        oh,
        ow,
        batch,
        col,
    );
}

/// Build the columns `[col_start, col_end)` of the im2col matrix for one
/// (batch, group) slice into a `[col_rows, col_end - col_start]` buffer.
///
/// Same gather as `im2col::im2col_adaptive`, restricted to a range of output
/// spatial positions and written with the *block* width as the row stride, so
/// the result feeds `matrixmultiply::sgemm` directly as a `[k, n]` operand.
/// Because im2col is a pure gather — every element is either an input element
/// or a padding zero — the block is bit-identical to the corresponding slice
/// of the monolithic matrix by construction.
///
/// The block boundaries are chosen as whole output rows whenever possible (see
/// [`im2col_block_cols`]); the head/tail arithmetic below is what makes an
/// unaligned block — a single output row wider than the workspace cap — work
/// as well.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_block(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    ow: usize,
    col_start: usize,
    col_end: usize,
    batch: usize,
    col: &mut [f32],
) {
    if ow == 0 || col_end <= col_start {
        return;
    }
    let block_w = col_end - col_start;
    let oy_first = col_start / ow;
    let oy_last = (col_end - 1) / ow;
    // Horizontal runs are contiguous in the input exactly when a step of one
    // output column is a step of one input column.
    let contiguous_rows = strides[1] == 1 && dilations[1] == 1;

    let mut row = 0;
    for ic in 0..c_per_group {
        let in_c = in_c_start + ic;
        let in_plane = &input[(batch * c_in + in_c) * h * w..][..h * w];
        for ky in 0..kh {
            for kx in 0..kw {
                let dst_row = &mut col[row * block_w..][..block_w];
                for oy in oy_first..=oy_last {
                    let row_base = oy * ow;
                    // Columns of this output row that lie inside the block.
                    let ox_start = col_start.saturating_sub(row_base);
                    let ox_end = (col_end - row_base).min(ow);
                    let dst_off = row_base + ox_start - col_start;
                    let seg = &mut dst_row[dst_off..dst_off + (ox_end - ox_start)];

                    let iy = (oy * strides[0] + ky * dilations[0]) as isize - pads[0] as isize;
                    if iy < 0 || iy >= h as isize {
                        // Whole segment is vertical padding.
                        seg.fill(0.0);
                        continue;
                    }
                    let src_row = &in_plane[iy as usize * w..][..w];

                    if contiguous_rows {
                        // ix = ox + kx - pad_left, valid for 0 <= ix < w. Both
                        // bounds are clamped into [ox_start, ox_end]: a
                        // segment can lie entirely inside the left or the
                        // right padding.
                        let lo = pads[1].saturating_sub(kx).clamp(ox_start, ox_end);
                        let hi = (w + pads[1]).saturating_sub(kx).clamp(lo, ox_end);
                        seg[..lo - ox_start].fill(0.0);
                        if lo < hi {
                            let src_start = lo + kx - pads[1];
                            seg[lo - ox_start..hi - ox_start]
                                .copy_from_slice(&src_row[src_start..src_start + (hi - lo)]);
                        }
                        seg[hi - ox_start..].fill(0.0);
                    } else {
                        for (t, dst) in seg.iter_mut().enumerate() {
                            let ix = ((ox_start + t) * strides[1] + kx * dilations[1]) as isize
                                - pads[1] as isize;
                            *dst = if ix >= 0 && ix < w as isize {
                                src_row[ix as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                }
                row += 1;
            }
        }
    }
}

/// Run [`im2col_block`], splitting the block by input channel across rayon
/// when the work justifies it — the blocked twin of
/// [`im2col_maybe_parallel`], and load-bearing: the convolutions that trip the
/// workspace cap are gather-bound (`inswapper_128`'s final `[3, 128, 7, 7]`
/// layer has `m = 3`, so the GEMM is noise next to the im2col).
///
/// Row `r` of the block belongs to input channel `r / (kH * kW)`, so a
/// contiguous band of input channels owns a contiguous band of rows: each
/// thread calls the same gather with a shifted `in_c_start` over a disjoint
/// sub-slice.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_block_maybe_parallel(
    input: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    in_c_start: usize,
    c_per_group: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    ow: usize,
    col_start: usize,
    col_end: usize,
    batch: usize,
    col: &mut [f32],
) {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        let block_w = col_end.saturating_sub(col_start);
        let num_threads = rayon::current_num_threads();
        let work = c_per_group * kh * kw * block_w;
        if num_threads > 1 && c_per_group >= 2 && work >= PARALLEL_GEMM_THRESHOLD {
            let chunk_ic = c_per_group.div_ceil(num_threads).max(1);
            let chunk_len = chunk_ic * kh * kw * block_w;
            if chunk_len > 0 {
                col.par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(t, sub)| {
                        let first = t * chunk_ic;
                        if first >= c_per_group {
                            return;
                        }
                        let count = (c_per_group - first).min(chunk_ic);
                        im2col_block(
                            input,
                            c_in,
                            h,
                            w,
                            in_c_start + first,
                            count,
                            kh,
                            kw,
                            strides,
                            pads,
                            dilations,
                            ow,
                            col_start,
                            col_end,
                            batch,
                            sub,
                        );
                    });
                return;
            }
        }
    }

    im2col_block(
        input,
        c_in,
        h,
        w,
        in_c_start,
        c_per_group,
        kh,
        kw,
        strides,
        pads,
        dilations,
        ow,
        col_start,
        col_end,
        batch,
        col,
    );
}

/// `C = A × B` with `beta = 0`, split by rows of `A` across rayon when large.
#[inline]
pub(crate) fn sgemm_maybe_parallel(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    sgemm_maybe_parallel_ldc(m, k, n, a, b, c, n);
}

/// [`sgemm_maybe_parallel`] with an explicit row stride for `C`.
///
/// `ldc == n` is the packed case and reproduces `sgemm_maybe_parallel`
/// exactly. `ldc > n` lets a column block of an im2col GEMM write straight
/// into the columns of a wider output matrix it owns, with no staging buffer
/// and no copy-back.
#[inline]
fn sgemm_maybe_parallel_ldc(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
) {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        let num_threads = rayon::current_num_threads();
        if m.saturating_mul(k).saturating_mul(n) >= PARALLEL_GEMM_THRESHOLD && m >= num_threads * 2
        {
            parallel_sgemm(m, k, n, a, b, c, ldc);
            return;
        }
    }
    sgemm_sequential_ldc(m, k, n, a, b, c, ldc);
}

/// One sequential `matrixmultiply::sgemm` call: `C = A × B`, `beta = 0`, with
/// an explicit row stride for `C`.
#[inline]
#[allow(unsafe_code)]
fn sgemm_sequential_ldc(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
    }
}

/// Single-job (batch=1, group=1) convolution: im2col → parallel GEMM → bias,
/// writing straight into `out` with the shared `col` scratch buffer.
///
/// Split out of [`conv2d_into_slices_capped`] so the caller can hand it a
/// cached thread-local scratch buffer (`COL_SCRATCH`) instead of allocating
/// (and zero-filling, and page-faulting in) a fresh ~56 MB column matrix for
/// every node. Arithmetic and accumulation order are exactly those of the
/// original inline loop, so results are bit-identical.
#[allow(clippy::too_many_arguments)]
fn single_job_parallel(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    c_in: usize,
    h: usize,
    w: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    n: usize,
    group: usize,
    c_out: usize,
    col_rows: usize,
    col_cols: usize,
    block_cols: usize,
    col: &mut [f32],
    out: &mut [f32],
) {
    for batch in 0..n {
        for g in 0..group {
            let in_c_start = g * c_per_group;
            let w_off = g * c_out_per_group * col_rows;
            let o_off = (batch * c_out + g * c_out_per_group) * col_cols;
            if block_cols == col_cols {
                im2col_maybe_parallel(
                    input,
                    c_in,
                    h,
                    w,
                    in_c_start,
                    c_per_group,
                    kh,
                    kw,
                    strides,
                    pads,
                    dilations,
                    oh,
                    ow,
                    batch,
                    col,
                );
                sgemm_maybe_parallel(
                    c_out_per_group,
                    col_rows,
                    col_cols,
                    &weight[w_off..],
                    col,
                    &mut out[o_off..],
                );
            } else {
                let mut col_start = 0;
                while col_start < col_cols {
                    let col_end = (col_start + block_cols).min(col_cols);
                    let block_w = col_end - col_start;
                    let col_block = &mut col[..col_rows * block_w];
                    im2col_block_maybe_parallel(
                        input,
                        c_in,
                        h,
                        w,
                        in_c_start,
                        c_per_group,
                        kh,
                        kw,
                        strides,
                        pads,
                        dilations,
                        ow,
                        col_start,
                        col_end,
                        batch,
                        col_block,
                    );
                    sgemm_maybe_parallel_ldc(
                        c_out_per_group,
                        col_rows,
                        block_w,
                        &weight[w_off..],
                        col_block,
                        &mut out[o_off + col_start..],
                        col_cols,
                    );
                    col_start = col_end;
                }
            }
            if let Some(b) = bias {
                for oc in 0..c_out_per_group {
                    let bv = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
                    let start = o_off + oc * col_cols;
                    for j in 0..col_cols {
                        out[start + j] += bv;
                    }
                }
            }
        }
    }
}

/// Process one (batch, group) slice: im2col → sgemm → bias, writing directly
/// into `dst` (`[c_out_per_group, col_cols]`, row-major) instead of
/// allocating and returning a fresh `Vec<f32>`.
///
/// `col_scratch` must have length `>= col_rows * block_cols`; it is reused
/// across calls by the caller (one buffer per rayon worker via
/// `for_each_init`, or one buffer for the whole wasm32 loop) rather than
/// allocated fresh per job. Safe to reuse dirty, since `im2col_adaptive` and
/// [`im2col_block`] unconditionally overwrite every element they use of
/// `col_scratch`, including explicit zero-fills for padding.
///
/// `block_cols == col_cols` means the whole column matrix fits the workspace
/// cap: the job runs as one im2col + one GEMM, exactly as before. Otherwise
/// the output columns are walked in blocks of `block_cols`, each GEMM writing
/// into the columns of `dst` it owns (row stride `col_cols`).
#[allow(clippy::too_many_arguments)]
fn conv2d_single_job_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    c_in: usize,
    h: usize,
    w: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    col_rows: usize,
    col_cols: usize,
    block_cols: usize,
    batch: usize,
    g: usize,
    col_scratch: &mut [f32],
    dst: &mut [f32],
) {
    let in_c_start = g * c_per_group;
    let w_off = g * c_out_per_group * col_rows;

    // Already inside a rayon job — keep the inner GEMMs sequential.
    if block_cols == col_cols {
        let col = &mut col_scratch[..col_rows * col_cols];
        im2col_adaptive(
            input,
            c_in,
            h,
            w,
            in_c_start,
            c_per_group,
            kh,
            kw,
            strides,
            pads,
            dilations,
            oh,
            ow,
            batch,
            col,
        );
        sgemm_sequential_ldc(
            c_out_per_group,
            col_rows,
            col_cols,
            &weight[w_off..],
            col,
            dst,
            col_cols,
        );
    } else {
        let mut col_start = 0;
        while col_start < col_cols {
            let col_end = (col_start + block_cols).min(col_cols);
            let block_w = col_end - col_start;
            let col_block = &mut col_scratch[..col_rows * block_w];
            im2col_block(
                input,
                c_in,
                h,
                w,
                in_c_start,
                c_per_group,
                kh,
                kw,
                strides,
                pads,
                dilations,
                ow,
                col_start,
                col_end,
                batch,
                col_block,
            );
            sgemm_sequential_ldc(
                c_out_per_group,
                col_rows,
                block_w,
                &weight[w_off..],
                col_block,
                &mut dst[col_start..],
                col_cols,
            );
            col_start = col_end;
        }
    }

    if let Some(b) = bias {
        for oc in 0..c_out_per_group {
            let bias_val = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
            let row_start = oc * col_cols;
            for j in 0..col_cols {
                dst[row_start + j] += bias_val;
            }
        }
    }
}

/// 1×1 convolution fast path, writing directly into a pre-allocated buffer.
///
/// For 1×1 kernel with stride=1, no padding, dilation=1:
///   input  [N, C_in, H, W]  → treat as [N, C_in, H*W]
///   weight [C_out, C_in/g, 1, 1] → treat as [C_out/g, C_in/g]
///   output = weight × input_slice (matmul, no copy)
///
/// Direct 3×3 convolution (no im2col).
///
/// Zero-pad an NHWC-agnostic `[n, c_in, h, w]` input for the direct conv.
/// Returns `(padded_data, padded_h, padded_w)`.
#[allow(clippy::too_many_arguments)]
fn pad_input_for_direct(
    input: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    pads: [usize; 4],
) -> (Vec<f32>, usize, usize) {
    let [pt, pl, pb, pr] = pads;
    let ph = h + pt + pb;
    let pw = w + pl + pr;
    let plane = h * w;
    let pplane = ph * pw;
    let mut out = vec![0.0f32; n * c_in * pplane];
    for ni in 0..n {
        for ci in 0..c_in {
            let src = (ni * c_in + ci) * plane;
            let dst = (ni * c_in + ci) * pplane + pt * pw + pl;
            for y in 0..h {
                out[dst + y * pw..dst + y * pw + w]
                    .copy_from_slice(&input[src + y * w..src + y * w + w]);
            }
        }
    }
    (out, ph, pw)
}

/// Direct small-M convolution: `c_out ≤ 3`, stride 1, dilation 1, group 1.
///
/// Computes `out[n][co][oy][ox] = bias[co] + Σ_{ci,ky,kx} in[n][ci][oy+ky][ox+kx] * w[co][ci][ky][kx]`
/// with no im2col materialization.
///
/// # Strategy
/// One output row at a time, vectorized 24-wide over `ox` (three ymm per
/// output channel). With `c_out = 3` that is 9 independent FMA chains,
/// enough to saturate both FMA ports despite the 4-cycle FMA latency.
/// The inner `(ci, ky, kx)` loop loads one 8-wide input vector per tap
/// and broadcasts a scalar weight per output channel.
///
/// # Why this beats im2col + GEMM at M=3
/// `matrixmultiply` measures ~11 GFLOP/s at M=3 (its MR=6 tile wastes half
/// the work and its packing overhead dominates), and an im2col for a 7×7
/// kernel over a 506×506 output allocates and streams ~3.2 GB. This kernel
/// touches each input element once per output row (cache-resident re-reads
/// for the kernel taps) and never materializes the column matrix.
///
/// # Safety
/// Requires AVX2+FMA (caller checks) and no padding (caller zero-pads).
/// `out` must hold at least `n * c_out * oh * ow` elements; `input` at
/// least `n * c_in * h * w`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv2d_direct_small_m_f32(
    input: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    out: &mut [f32],
) {
    use core::arch::x86_64::*;

    let in_plane = h * w;
    let out_plane = oh * ow;
    let taps = c_in * kh * kw;

    let mut bias_vals = [0.0f32; 3];
    for (co, slot) in bias_vals.iter_mut().enumerate().take(c_out) {
        *slot = bias.map(|b| b[co]).unwrap_or(0.0);
    }

    // 24-wide main loop + scalar tail. Blocks of 8 for the tail remainder.
    let n_blocks24 = ow / 24;
    let n_blocks8 = (ow % 24) / 8;
    let tail_start = n_blocks24 * 24 + n_blocks8 * 8;

    for ni in 0..n {
        let in_batch = ni * c_in * in_plane;
        let out_batch = ni * c_out * out_plane;

        for oy in 0..oh {
            let out_row = out_batch + oy * ow;

            // ── 24-wide blocks ────────────────────────────────────────────
            for blk in 0..n_blocks24 {
                let ox = blk * 24;
                let mut acc0 = [_mm256_setzero_ps(); 3]; // co 0, ox..ox+7 / +8..+15 / +16..+23
                let mut acc1 = [_mm256_setzero_ps(); 3];
                let mut acc2 = [_mm256_setzero_ps(); 3];

                for ci in 0..c_in {
                    let in_ch = in_batch + ci * in_plane;
                    for ky in 0..kh {
                        let row = in_ch + (oy + ky) * w + ox;
                        for kx in 0..kw {
                            let v0 = _mm256_loadu_ps(input.as_ptr().add(row + kx));
                            let v1 = _mm256_loadu_ps(input.as_ptr().add(row + kx + 8));
                            let v2 = _mm256_loadu_ps(input.as_ptr().add(row + kx + 16));
                            let w_base = ci * c_out * kh * kw + ky * kw + kx;
                            // co = 0
                            if c_out > 0 {
                                let ws = _mm256_broadcast_ss(&weight[w_base]);
                                acc0[0] = _mm256_fmadd_ps(v0, ws, acc0[0]);
                                acc0[1] = _mm256_fmadd_ps(v1, ws, acc0[1]);
                                acc0[2] = _mm256_fmadd_ps(v2, ws, acc0[2]);
                            }
                            if c_out > 1 {
                                let ws = _mm256_broadcast_ss(
                                    &weight[w_base + kh * kw],
                                );
                                acc1[0] = _mm256_fmadd_ps(v0, ws, acc1[0]);
                                acc1[1] = _mm256_fmadd_ps(v1, ws, acc1[1]);
                                acc1[2] = _mm256_fmadd_ps(v2, ws, acc1[2]);
                            }
                            if c_out > 2 {
                                let ws = _mm256_broadcast_ss(
                                    &weight[w_base + 2 * kh * kw],
                                );
                                acc2[0] = _mm256_fmadd_ps(v0, ws, acc2[0]);
                                acc2[1] = _mm256_fmadd_ps(v1, ws, acc2[1]);
                                acc2[2] = _mm256_fmadd_ps(v2, ws, acc2[2]);
                            }
                        }
                    }
                }

                // Add bias and store (channel stride is `out_plane`)
                if c_out > 0 {
                    let bs = _mm256_set1_ps(bias_vals[0]);
                    let ptr = out.as_mut_ptr().add(out_row + ox);
                    _mm256_storeu_ps(ptr, _mm256_add_ps(acc0[0], bs));
                    _mm256_storeu_ps(ptr.add(8), _mm256_add_ps(acc0[1], bs));
                    _mm256_storeu_ps(ptr.add(16), _mm256_add_ps(acc0[2], bs));
                }
                if c_out > 1 {
                    let bs = _mm256_set1_ps(bias_vals[1]);
                    let ptr = out
                        .as_mut_ptr()
                        .add(out_batch + out_plane + oy * ow + ox);
                    _mm256_storeu_ps(ptr, _mm256_add_ps(acc1[0], bs));
                    _mm256_storeu_ps(ptr.add(8), _mm256_add_ps(acc1[1], bs));
                    _mm256_storeu_ps(ptr.add(16), _mm256_add_ps(acc1[2], bs));
                }
                if c_out > 2 {
                    let bs = _mm256_set1_ps(bias_vals[2]);
                    let ptr = out
                        .as_mut_ptr()
                        .add(out_batch + 2 * out_plane + oy * ow + ox);
                    _mm256_storeu_ps(ptr, _mm256_add_ps(acc2[0], bs));
                    _mm256_storeu_ps(ptr.add(8), _mm256_add_ps(acc2[1], bs));
                    _mm256_storeu_ps(ptr.add(16), _mm256_add_ps(acc2[2], bs));
                }
            }

            // ── 8-wide blocks ─────────────────────────────────────────────
            for blk in 0..n_blocks8 {
                let ox = n_blocks24 * 24 + blk * 8;
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();

                for ci in 0..c_in {
                    let in_ch = in_batch + ci * in_plane;
                    for ky in 0..kh {
                        let row = in_ch + (oy + ky) * w + ox;
                        for kx in 0..kw {
                            let v0 = _mm256_loadu_ps(input.as_ptr().add(row + kx));
                            let w_base = ci * c_out * kh * kw + ky * kw + kx;
                            if c_out > 0 {
                                let ws = _mm256_broadcast_ss(&weight[w_base]);
                                acc0 = _mm256_fmadd_ps(v0, ws, acc0);
                            }
                            if c_out > 1 {
                                let ws = _mm256_broadcast_ss(&weight[w_base + kh * kw]);
                                acc1 = _mm256_fmadd_ps(v0, ws, acc1);
                            }
                            if c_out > 2 {
                                let ws = _mm256_broadcast_ss(&weight[w_base + 2 * kh * kw]);
                                acc2 = _mm256_fmadd_ps(v0, ws, acc2);
                            }
                        }
                    }
                }

                if c_out > 0 {
                    let s0 = _mm256_add_ps(acc0, _mm256_set1_ps(bias_vals[0]));
                    _mm256_storeu_ps(out.as_mut_ptr().add(out_batch + oy * ow + ox), s0);
                }
                if c_out > 1 {
                    let s1 = _mm256_add_ps(acc1, _mm256_set1_ps(bias_vals[1]));
                    _mm256_storeu_ps(
                        out.as_mut_ptr()
                            .add(out_batch + out_plane + oy * ow + ox),
                        s1,
                    );
                }
                if c_out > 2 {
                    let s2 = _mm256_add_ps(acc2, _mm256_set1_ps(bias_vals[2]));
                    _mm256_storeu_ps(
                        out.as_mut_ptr()
                            .add(out_batch + 2 * out_plane + oy * ow + ox),
                        s2,
                    );
                }
            }

            // ── Scalar tail ───────────────────────────────────────────────
            for ox in tail_start..ow {
                for co in 0..c_out {
                    let mut acc = bias_vals[co];
                    for ci in 0..c_in {
                        let in_ch = in_batch + ci * in_plane;
                        for ky in 0..kh {
                            let row = in_ch + (oy + ky) * w + ox;
                            for kx in 0..kw {
                                acc += input[row + kx]
                                    * weight[ci * c_out * kh * kw + co * kh * kw + ky * kw + kx];
                            }
                        }
                    }
                    out[out_batch + co * out_plane + oy * ow + ox] = acc;
                }
            }
        }
    }
    let _ = taps;
}

/// This saves allocating and filling the im2col column matrix.
#[allow(clippy::too_many_arguments)]
fn conv2d_1x1_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    c_in: usize,
    c_out: usize,
    c_per_group: usize,
    c_out_per_group: usize,
    h: usize,
    w: usize,
    group: usize,
    out: &mut [f32],
) {
    let spatial = h * w;

    for batch in 0..n {
        for g in 0..group {
            let in_c_start = g * c_per_group;
            let in_off = (batch * c_in + in_c_start) * spatial;
            let w_off = g * c_out_per_group * c_per_group;
            let o_off = (batch * c_out + g * c_out_per_group) * spatial;

            sgemm_maybe_parallel(
                c_out_per_group,
                c_per_group,
                spatial,
                &weight[w_off..],
                &input[in_off..],
                &mut out[o_off..],
            );

            if let Some(b) = bias {
                for oc in 0..c_out_per_group {
                    let bv = b.get(g * c_out_per_group + oc).copied().unwrap_or(0.0_f32);
                    let start = o_off + oc * spatial;
                    for j in 0..spatial {
                        out[start + j] += bv;
                    }
                }
            }
        }
    }
}

/// Split a large sgemm (C = A × B) by rows of A across rayon threads.
/// A: [m, k] row-major, B: [k, n] row-major, C: [m, >= n] row-major.
///
/// Each thread runs the *same* `matrixmultiply::sgemm` over a disjoint row
/// band, and the K-blocking (hence the accumulation order of every output
/// element) does not depend on `m`, so the result is bit-identical to one
/// sequential call — asserted by `parallel_sgemm_is_bitwise_identical`.
///
/// `ldc` is the row stride of `C`; `ldc == n` is the packed case. The row
/// bands stay disjoint whatever the stride: band `t` owns rows
/// `[t * chunk, (t + 1) * chunk)`, and a band's slice may cover the columns
/// beyond `n` of its own last row — which this GEMM never writes, and which
/// belong to a different column block of the *same* job, processed before or
/// after this call, never concurrently.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub(crate) fn parallel_sgemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 || ldc == 0 {
        return;
    }
    let num_threads = rayon::current_num_threads();

    // ── Split N when it can feed every thread ─────────────────────────────
    //
    // `matrixmultiply` packs B (the `[k, n]` operand) into cache blocks on
    // every call. Splitting M means every thread packs the *same* B — for the
    // FFC workhorse GEMM (`m=128, k=3456, n=4096`) that is a 56 MB pack
    // repeated 12 times, ~672 MB of duplicate traffic, and it measured at
    // 32 ms against ~5 ms of useful FMA work. Splitting N gives each thread a
    // disjoint column range of B and C, so B is packed exactly once in
    // total and A (the small operand) once per thread.
    //
    // Bit-identical: the K-accumulation order of each output element is fixed
    // by the microkernel and does not depend on how N is split — the same
    // property the column-blocked `_ldc` path already relies on.
    let n_chunk = n.div_ceil(num_threads.max(1)).max(1);
    if n >= num_threads * 32 {
        let b_ptr = b.as_ptr() as usize;
        let c_ptr = c.as_mut_ptr() as usize;
        (0..n)
            .into_par_iter()
            .step_by(n_chunk)
            .for_each(|n_start| {
                let block_n = n_chunk.min(n - n_start);
                // SAFETY: each task owns the disjoint column range
                // `[n_start, n_start + block_n)` of both B (read) and C
                // (write). B is `[k, n]` row-major with row stride `n`, so the
                // sub-matrix starting at column `n_start` is described by
                // `rsb = n, csb = 1`. C is `[m, ldc]` with its column range at
                // offset `n_start` (`rsc = ldc, csc = 1`).
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        block_n,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        (b_ptr as *const f32).add(n_start),
                        n as isize,
                        1,
                        0.0,
                        (c_ptr as *mut f32).add(n_start),
                        ldc as isize,
                        1,
                    );
                }
            });
        return;
    }

    // ── Fall back to the M-split when N is too small to share ─────────────
    let chunk = m.div_ceil(num_threads.max(1)).max(1);

    // `par_chunks_mut` hands each thread a disjoint, correctly-sized row band
    // of C, so the tiles are written in place — no scratch allocation and no
    // second pass over the result.
    c[..(m - 1) * ldc + n]
        .par_chunks_mut(chunk * ldc)
        .enumerate()
        .for_each(|(t, tile)| {
            let row_start = t * chunk;
            if row_start >= m {
                return;
            }
            let tile_m = (m - row_start).min(chunk);
            // SAFETY: `a` is [m, k] row-major so row `row_start` starts at
            // `row_start * k`; `tile` starts at row `row_start` of C and spans
            // `(tile_m - 1) * ldc + n` elements or more — full chunks are
            // `chunk * ldc >= (chunk - 1) * ldc + n` since `ldc >= n`, and the
            // final short chunk ends exactly at the last element written.
            unsafe {
                matrixmultiply::sgemm(
                    tile_m,
                    k,
                    n,
                    1.0,
                    a[row_start * k..].as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    tile.as_mut_ptr(),
                    ldc as isize,
                    1,
                );
            }
        });
}
