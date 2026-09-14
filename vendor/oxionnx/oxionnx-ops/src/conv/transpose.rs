// Modified by the GIMP LaMa inpainting fork of OxiONNX (2026-09): ADDS row-wise (and 2-channel) AVX2 ConvTranspose kernels for stride 2 / kernel 3.
// See ../../../MODIFICATIONS.md for the full change list and rationale.

//! Rank-generic transposed convolution (`ConvTranspose1D/2D/3D`).
//!
//! A transposed convolution is a scatter-accumulate: every input element
//! contributes `in_val * weight[k]` to the output position
//! `i * stride + k * dilation - pad_begin`, dropped when that lands outside
//! the (cropped) output extent.
//!
//! The rank-2 loop nest is kept as a specialisation because it is the shape
//! decoders actually use; rank 1 and rank ≥ 3 run the generic nest. The two
//! visit output elements in the same order — batch, group, input channel,
//! input position, output channel, kernel offset — so their floating-point
//! accumulation is bit-identical, which
//! `conv_transpose_generic_matches_rank2_bitwise` asserts directly.

use oxionnx_core::{OnnxError, Tensor};

use super::spatial::{self, odometer_next};

/// Validated spatial parameters of an N-D transposed convolution.
pub(crate) struct ConvTransposeParams<'a> {
    /// One stride per spatial axis.
    pub strides: &'a [usize],
    /// `[begin_0, …, begin_{r-1}, end_0, …, end_{r-1}]`.
    pub pads: &'a [usize],
    /// One dilation per spatial axis.
    pub dilations: &'a [usize],
    /// Channel-group count.
    pub group: usize,
}

/// Compute output shape for transposed 2D convolution.
///
/// `input_shape`:  `[N, C_in, H, W]`
/// `weight_shape`: `[C_in, C_out/group, kH, kW]`
/// `pads`:         `[top, left, bottom, right]`
///
/// Returns `[N, C_out, oH, oW]` where `C_out = weight_shape[1] * group`.
///
/// Every size computation is checked: a malformed model (rank mismatch,
/// zero-size input, zero stride/dilation, `group == 0`, or padding that
/// exceeds the natural output extent) yields a typed
/// [`OnnxError::ShapeMismatch`] instead of an unsigned underflow.
pub(crate) fn compute_conv_transpose2d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    output_padding: &[usize],
    dilations: &[usize],
    group: usize,
) -> Result<Vec<usize>, OnnxError> {
    if input_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "ConvTranspose: input must be 4D [N,C,H,W], got rank {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch(format!(
            "ConvTranspose: weight must be 4D [C_in,C_out/group,kH,kW], got rank {}",
            weight_shape.len()
        )));
    }
    spatial::compute_conv_transpose_out_shape(
        "ConvTranspose",
        input_shape,
        weight_shape,
        strides,
        pads,
        output_padding,
        dilations,
        group,
    )
}

/// Rank-generic transposed convolution into a pre-allocated buffer.
///
/// `out` is zeroed before accumulation begins. `out_shape` must be the shape
/// [`spatial::compute_conv_transpose_out_shape`] derives for the same
/// parameters — the padding is *not* re-applied here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose_into(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<(), OnnxError> {
    let op = "ConvTranspose";
    let rank = spatial::spatial_rank(input_shape, op, "input")?;
    if weight_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: weight rank {} must equal input rank {} ([C_in, C_out/group, k_0, ...])",
            weight_shape.len(),
            input_shape.len()
        )));
    }
    if out_shape.len() != input_shape.len() {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: output rank {} must equal input rank {}",
            out_shape.len(),
            input_shape.len()
        )));
    }
    if params.group == 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: group must be >= 1, got 0"
        )));
    }
    if params.strides.len() != rank || params.dilations.len() != rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides ({}) and dilations ({}) need {rank} entries",
            params.strides.len(),
            params.dilations.len()
        )));
    }
    if params.pads.len() != 2 * rank {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: pads needs {} entries, got {}",
            2 * rank,
            params.pads.len()
        )));
    }
    if params.strides.contains(&0) || params.dilations.contains(&0) {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: strides and dilations must be >= 1"
        )));
    }
    let c_in = input_shape[1];
    if c_in != weight_shape[0] {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: input channels {c_in} != weight input channels {}",
            weight_shape[0]
        )));
    }
    if c_in % params.group != 0 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: c_in ({c_in}) not divisible by group ({})",
            params.group
        )));
    }
    let c_out_per_group = weight_shape[1];
    let c_out = c_out_per_group
        .checked_mul(params.group)
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: output channel count overflows")))?;
    if out_shape[1] != c_out {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: output channels {} != weight output channels {c_out_per_group} * group {}",
            out_shape[1], params.group
        )));
    }
    let in_len = volume(input_shape, op, "input")?;
    let w_len = volume(weight_shape, op, "weight")?;
    let out_len = volume(out_shape, op, "output")?;
    if input.len() < in_len || weight.len() < w_len || out.len() < out_len {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: buffer too small for its shape (input {} < {in_len}, weight {} < {w_len}, \
             output {} < {out_len})",
            input.len(),
            weight.len(),
            out.len()
        )));
    }
    if let Some(b) = bias {
        if b.len() < c_out {
            return Err(OnnxError::ShapeMismatch(format!(
                "{op}: bias has {} entries, expected {c_out}",
                b.len()
            )));
        }
    }

    out.fill(0.0_f32);
    if out_len == 0 || in_len == 0 {
        return Ok(());
    }

    if rank == 2 {
        // Fast path 1: direct ConvTranspose for stride=2, kernel=3, pad=1.
        // This avoids both the scatter-accumulate (terrible cache behavior) and
        // the zero-pad + full Conv2d (wastes FLOPs on zeros). Each output pixel
        // reads from at most 4 input pixels. Parallelized over output channels.
        if let Some(result) = try_convtranspose_direct_s2k3(
            input, input_shape, weight, weight_shape, params, out, out_shape,
        ) {
            result?;
        } else if let Some(result) = try_convtranspose_as_conv(
            input, input_shape, weight, weight_shape, params, out, out_shape,
        ) {
            result?;
        } else {
            scatter_rank2(
                input,
                input_shape,
                weight,
                weight_shape,
                params,
                out,
                out_shape,
            );
        }
    } else {
        scatter_generic(
            input,
            input_shape,
            weight,
            weight_shape,
            params,
            out,
            out_shape,
        );
    }

    if let Some(b) = bias {
        let n = out_shape[0];
        let out_plane: usize = out_shape[2..].iter().product();
        for ni in 0..n {
            for (co, &bias_val) in b.iter().take(c_out).enumerate() {
                let base = (ni * c_out + co) * out_plane;
                for v in &mut out[base..base + out_plane] {
                    *v += bias_val;
                }
            }
        }
    }
    Ok(())
}

/// Direct ConvTranspose2D for stride=2, kernel=3, pad=1, dilation=1, group=1.
///
/// Instead of scatter-accumulate (terrible cache behavior) or zero-pad +
/// full Conv2d (wastes FLOPs on zero-multiplication), this writes each output
/// pixel directly by reading from the ≤4 nearest input pixels. For stride=2,
/// kernel=3, each output pixel (oy, ox) reads from input pixels at positions
/// determined by the parity of oy and ox.
///
/// Parallelized over output channels with rayon. Each output channel plane
/// is independent, so the parallelism is trivially safe.
///
/// Returns `Some(Ok(()))` on success, `Some(Err(...))` on shape mismatch,
/// `None` if the parameters don't match this fast path.
#[allow(clippy::too_many_arguments)]
fn try_convtranspose_direct_s2k3(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) -> Option<Result<(), OnnxError>> {
    // Only handle: 4D, stride=2, kernel=3, dilation=1, group=1, pads=[1,1,1,1]
    if input_shape.len() != 4
        || weight_shape.len() != 4
        || params.strides != [2, 2]
        || params.dilations != [1, 1]
        || params.group != 1
        || params.pads.len() != 4
        || params.pads != [1, 1, 1, 1]
        || weight_shape[2] != 3
        || weight_shape[3] != 3
    {
        return None;
    }

    let (n, c_in, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let c_out = weight_shape[1]; // group=1
    let (oh, ow) = (out_shape[2], out_shape[3]);

    // For stride=2, kernel=3, pad=1, output_padding=1: oh = 2*h, ow = 2*w
    // We accept oh <= 2*h, ow <= 2*w and crop the excess.
    if oh > 2 * h || ow > 2 * w {
        return None;
    }

    let in_plane = h * w;
    let out_plane = oh * ow;

    // Weight stride: weight[ci][co][ky][kx] = weight[((ci*c_out + co)*3 + ky)*3 + kx]
    // For fixed co, varying ci: offset = ci * (c_out*9) + co*9 + ky*3 + kx
    let w_ci_stride = c_out * 9;

    // Parallelize over output planes (two planes per task when possible).
    //
    // A row-block parallelisation was tried and measured *slower* (387 ms vs
    // 354 ms node total): on this CPU (i7-13620H, 24 MB L3) the 8.4 MB input
    // already stays resident across the 256 fine-grained plane tasks, so the
    // 16 coarse row-block tasks bought no memory reuse and cost load balance
    // (16 tasks of unequal length against a 6P+4E core topology).
    //
    // Pairing output channels halves the input traffic — each channel reads
    // the whole input once, so the node streams ~6 GB of L3/DRAM. The paired
    // kernel is chosen only when AVX2+FMA is present and the batch/channel
    // counts pair cleanly; otherwise the single-channel loop below runs.
    #[cfg(target_arch = "x86_64")]
    let use_pairs =
        n == 1 && c_out % 2 == 0 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_pairs = false;

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        use rayon::prelude::*;
        if use_pairs {
            out.par_chunks_mut(2 * out_plane)
                .enumerate()
                .for_each(|(pair, slice)| {
                    let (plane_a, plane_b) = slice.split_at_mut(out_plane);
                    // Output channels of this pair are `2*pair` and `2*pair+1`;
                    // each channel's 9 weights start at `co * 9` with stride
                    // `c_out * 9` between input channels.
                    let w_co_base = pair * 2 * 9;
                    // SAFETY: AVX2+FMA confirmed above; the slices own the two
                    // planes of this pair, disjoint from every other task.
                    unsafe {
                        convtranspose_s2k3_plane2_rowwise(
                            input,
                            weight,
                            0,
                            w_co_base,
                            c_in,
                            h,
                            w,
                            oh,
                            ow,
                            in_plane,
                            w_ci_stride,
                            0,
                            oh,
                            plane_a,
                            plane_b,
                        );
                    }
                });
        } else {
            out.par_chunks_mut(out_plane)
                .enumerate()
                .for_each(|(idx, out_slice)| {
                    let ni = idx / c_out;
                    let co = idx % c_out;
                    convtranspose_s2k3_plane_rows(
                        input, weight, ni, co, c_in, h, w, oh, ow, in_plane, w_ci_stride, 0, oh,
                        out_slice,
                    );
                });
        }
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        for ni in 0..n {
            for co in 0..c_out {
                let o_base = (ni * c_out + co) * out_plane;
                convtranspose_s2k3_plane(
                    input,
                    weight,
                    ni,
                    co,
                    c_in,
                    h,
                    w,
                    oh,
                    ow,
                    in_plane,
                    w_ci_stride,
                    &mut out[o_base..o_base + out_plane],
                );
            }
        }
    }

    Some(Ok(()))
}

/// Compute one output plane [oh, ow] for one (batch, output_channel).
/// Inner loop of `try_convtranspose_direct_s2k3`.
///
/// For ConvTranspose(s=2, k=3, p=1), output(oy, ox) = sum over valid
/// (iy, ix, ky, kx) where oy = iy*2 + ky - 1 and ox = ix*2 + kx - 1.
/// For a given (oy, ox), valid ky ∈ {0,1,2} satisfy (oy+1-ky) even, and
/// similarly for kx. Since ky ∈ {0,1,2}, the valid values are:
///   ky = (oy+1) % 2  and  ky = (oy+1) % 2 + 2
/// So each output pixel has at most 2×2 = 4 contributions.
///
/// # AVX2 inline assembly optimization
///
/// Instead of the channel-dot-product approach (strided loads), we
/// vectorize across the output x-axis. For stride=2, kernel=3, the
/// even and odd output positions have complementary tap patterns:
///
/// - py=1, even ox: 1 tap (kx=1, ix=ox/2) — contiguous input at ix
/// - py=1, odd ox:  2 taps (kx=0, ix=(ox+1)/2; kx=2, ix=(ox-1)/2)
/// - py=0, even ox: 2 taps (kx=1, ix=ox/2 from two iy rows)
/// - py=0, odd ox:  4 taps (all four combinations)
///
/// For 4 consecutive same-parity outputs, the input positions are
/// CONTIGUOUS (ix increments by 1). This enables `vmovups` loads and
/// `vbroadcastss` weight broadcasts, with `vfmadd231ps` accumulating
/// 4 outputs per instruction.
///
/// The inline assembly loop processes one output row at a time, using
/// ymm registers for 8-wide FMA (4 even + 4 odd outputs interleaved).
///
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // only referenced by the wasm single-thread fallback
fn convtranspose_s2k3_plane(
    input: &[f32],
    weight: &[f32],
    ni: usize,
    co: usize,
    c_in: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    in_plane: usize,
    w_ci_stride: usize,
    out_slice: &mut [f32],
) {
    convtranspose_s2k3_plane_rows(
        input, weight, ni, co, c_in, h, w, oh, ow, in_plane, w_ci_stride, 0, oh, out_slice,
    );
}

/// [`convtranspose_s2k3_plane`] restricted to output rows `[oy0, oy1)`.
///
/// The row kernel is per-row independent, so a sub-range is exactly the
/// corresponding rows of the full-plane result. Used by the row-block-parallel
/// driver so each task's input slice stays cache-resident across planes.
#[allow(clippy::too_many_arguments)]
fn convtranspose_s2k3_plane_rows(
    input: &[f32],
    weight: &[f32],
    ni: usize,
    co: usize,
    c_in: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    in_plane: usize,
    w_ci_stride: usize,
    oy0: usize,
    oy1: usize,
    out_slice: &mut [f32],
) {
    let in_batch_base = ni * c_in * in_plane;
    let w_co_base = co * 9;

    #[cfg(target_arch = "x86_64")]
    let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;

    if avx2 {
        // SAFETY: AVX2+FMA confirmed via `is_x86_feature_detected!` above.
        unsafe {
            convtranspose_s2k3_plane_rowwise(
                input,
                weight,
                in_batch_base,
                w_co_base,
                c_in,
                h,
                w,
                oh,
                ow,
                in_plane,
                w_ci_stride,
                oy0,
                oy1,
                out_slice,
            );
        }
    } else {
        convtranspose_s2k3_plane_scalar(
            input,
            weight,
            in_batch_base,
            w_co_base,
            c_in,
            h,
            w,
            oh,
            ow,
            in_plane,
            w_ci_stride,
            oy0,
            oy1,
            out_slice,
        );
    }
}

/// Row-wise AVX2 ConvTranspose(s=2, k=3, p=1) kernel.
///
/// The per-pixel gather kernel (`convtranspose_s2k3_plane_avx2_asm`) uses
/// `vgatherdps` over channels — 2 gathers + 1 FMA per 8 channels per tap —
/// because consecutive output pixels read non-consecutive input positions.
/// Splitting even/odd output positions apart makes them *consecutive*: for
/// a fixed output row and parity class, output `ox = 2i + r` reads input
/// `ix = ix0 + i`. This kernel therefore processes 16 output columns (8 even
/// + 8 odd) per iteration with plain `vmovups` loads and 3-6 FMAs per input
/// channel — ~16x fewer instructions than the gather version.
///
/// Tap tables for stride 2, kernel 3, pad 1 (output `oy`/`ox` parity `py`/`px`):
///   even row (`ky=1`, input `iy=oy/2`):
///     even ox (`kx=1`): `ix=(ox)/2`;  odd ox: `kx=0 → ix=(ox+1)/2`, `kx=2 → ix=(ox-1)/2`
///   odd row: `ky=0` reads input `iy=(oy+1)/2` (loaded as `v`), `ky=2` reads
///    `iy=(oy-1)/2` (loaded as `u`); same column pattern on both rows, with
///    the ky=0 weights (`+0..+2`) on `v` and ky=2 (`+6..+8`) on `u`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn convtranspose_s2k3_plane_rowwise(
    input: &[f32],
    weight: &[f32],
    in_batch_base: usize,
    w_co_base: usize,
    c_in: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    in_plane: usize,
    w_ci_stride: usize,
    oy0: usize,
    oy1: usize,
    out_slice: &mut [f32],
) {
    use core::arch::x86_64::*;

    let in_ptr = input.as_ptr();
    let w_ptr = weight.as_ptr();
    let out_ptr = out_slice.as_mut_ptr();
    let _ = oh;

    // Shuffle indices for the masked "load at ix0+1" of the final block:
    // lane i gets element i+1; lane 7 is replaced by zero via the blend.
    let shift_idx = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);

    for oy in oy0..oy1 {
        let even_row = oy % 2 == 0;
        let ow_row = oy * ow;

        // Bottom edge: on the last odd row the ky=0 input row `(oy+1)/2` is
        // `h`, which does not exist. The SIMD path would read past the plane
        // and use it as if it were real, so that one row goes through the
        // bounds-checking scalar tail instead.
        if !even_row && (oy + 1) / 2 >= h {
            convtranspose_row_tail(
                input,
                weight,
                in_batch_base,
                w_co_base,
                c_in,
                h,
                w,
                ow,
                in_plane,
                w_ci_stride,
                oy,
                0,
                false,
                out_slice,
            );
            continue;
        }

        // Full 16-wide blocks. A block needs input columns [ix0, ix0+9) for
        // the plain loads; the final block (ix0 + 8 == w) uses a shuffled
        // + blended load instead, because the one-past-the-row element it
        // would need only feeds output column `2w-1`'s `kx=0` tap, which the
        // spec requires to be dropped (input column `w` does not exist).
        let mut ix0 = 0usize;
        while ix0 + 8 <= w && ix0 * 2 + 16 <= ow {
            let last = ix0 + 8 == w;
            let mut acc_e = _mm256_setzero_ps();
            let mut acc_o1 = _mm256_setzero_ps();
            let mut acc_o2 = _mm256_setzero_ps();
            let mut acc_o3 = _mm256_setzero_ps();
            let mut acc_o4 = _mm256_setzero_ps();

            let (iy_a, iy_b) = if even_row {
                let iy = oy / 2;
                (iy, iy)
            } else {
                ((oy + 1) / 2, (oy - 1) / 2)
            };

            for ci in 0..c_in {
                let in_a = in_ptr.add(in_batch_base + ci * in_plane + iy_a * w + ix0);
                let v0 = _mm256_loadu_ps(in_a);
                let v1 = if last {
                    let shifted = _mm256_permutevar8x32_ps(v0, shift_idx);
                    _mm256_blend_ps(shifted, _mm256_setzero_ps(), 0x80)
                } else {
                    _mm256_loadu_ps(in_a.add(1))
                };

                let wb = w_ptr.add(w_co_base + ci * w_ci_stride);
                if even_row {
                    // weights at [ky=1]: kx=0/1/2 => offsets +3,+4,+5
                    let w10 = _mm256_broadcast_ss(&*wb.add(3));
                    let w11 = _mm256_broadcast_ss(&*wb.add(4));
                    let w12 = _mm256_broadcast_ss(&*wb.add(5));
                    acc_e = _mm256_fmadd_ps(v0, w11, acc_e);
                    acc_o1 = _mm256_fmadd_ps(v1, w10, acc_o1);
                    acc_o2 = _mm256_fmadd_ps(v0, w12, acc_o2);
                } else {
                    let in_b = in_ptr.add(in_batch_base + ci * in_plane + iy_b * w + ix0);
                    let u0 = _mm256_loadu_ps(in_b);
                    let u1 = if last {
                        let shifted = _mm256_permutevar8x32_ps(u0, shift_idx);
                        _mm256_blend_ps(shifted, _mm256_setzero_ps(), 0x80)
                    } else {
                        _mm256_loadu_ps(in_b.add(1))
                    };

                    // Odd rows: `v` is the ky=0 input row ((oy+1)/2), `u` the
                    // ky=2 row ((oy-1)/2). Pair each row with its own kernel
                    // rows: ky=0 weights (+0..+2) for v, ky=2 (+6..+8) for u.
                    let w00 = _mm256_broadcast_ss(&*wb.add(0));
                    let w01 = _mm256_broadcast_ss(&*wb.add(1));
                    let w02 = _mm256_broadcast_ss(&*wb.add(2));
                    let w20 = _mm256_broadcast_ss(&*wb.add(6));
                    let w21 = _mm256_broadcast_ss(&*wb.add(7));
                    let w22 = _mm256_broadcast_ss(&*wb.add(8));

                    acc_e = _mm256_fmadd_ps(u0, w21, acc_e);
                    acc_e = _mm256_fmadd_ps(v0, w01, acc_e);
                    acc_o1 = _mm256_fmadd_ps(u1, w20, acc_o1);
                    acc_o2 = _mm256_fmadd_ps(u0, w22, acc_o2);
                    acc_o3 = _mm256_fmadd_ps(v1, w00, acc_o3);
                    acc_o4 = _mm256_fmadd_ps(v0, w02, acc_o4);
                }
            }

            // Even rows: acc_o = acc_o1 + acc_o2.
            // Odd rows:  acc_o = acc_o1 + acc_o2 + acc_o3 + acc_o4.
            let acc_o = if even_row {
                _mm256_add_ps(acc_o1, acc_o2)
            } else {
                _mm256_add_ps(_mm256_add_ps(acc_o1, acc_o2), _mm256_add_ps(acc_o3, acc_o4))
            };

            // Interleave even/odd accumulators into 16 consecutive outputs.
            let lo = _mm256_unpacklo_ps(acc_e, acc_o);
            let hi = _mm256_unpackhi_ps(acc_e, acc_o);
            let r0 = _mm256_permute2f128_ps(lo, hi, 0x20);
            let r1 = _mm256_permute2f128_ps(lo, hi, 0x31);
            _mm256_storeu_ps(out_ptr.add(ow_row + ix0 * 2), r0);
            _mm256_storeu_ps(out_ptr.add(ow_row + ix0 * 2 + 8), r1);

            ix0 += 8;
        }

        // Scalar tail for the remaining columns (rare: `w % 8 != 0`).
        if ix0 * 2 < ow {
            convtranspose_row_tail(
                input,
                weight,
                in_batch_base,
                w_co_base,
                c_in,
                h,
                w,
                ow,
                in_plane,
                w_ci_stride,
                oy,
                ix0 * 2,
                even_row,
                out_slice,
            );
        }
    }
}

/// Two-output-channel variant of [`convtranspose_s2k3_plane_rowwise`].
///
/// Every output channel reads the entire input once — 24 MB per plane and
/// 6.1 GB across the 256 planes of `model.24` — so the node is bandwidth
/// bound long before it is FMA bound. Processing two channels per pass over
/// the input halves that traffic (the input vectors `v0`/`v1`/`u0`/`u1` are
/// loaded once and FMA'd into both channels' accumulators). Per-element
/// accumulation order is unchanged, so the result is bit-identical.
///
/// Register budget: 6 accumulators (even rows) or 10 (odd rows), plus the
/// 2-4 input vectors and one broadcast temp — within the 16 ymm available.
///
/// # Safety
/// AVX2+FMA must be available (caller checks). `out_a` and `out_b` must each
/// hold `oh * ow` elements; `weight` must contain both channels' kernels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn convtranspose_s2k3_plane2_rowwise(
    input: &[f32],
    weight: &[f32],
    in_batch_base: usize,
    w_co_base: usize,
    c_in: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    in_plane: usize,
    w_ci_stride: usize,
    oy0: usize,
    oy1: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    use core::arch::x86_64::*;
    let _ = (oh, h);
    let in_ptr = input.as_ptr();
    let w_ptr = weight.as_ptr();
    let out_a_ptr = out_a.as_mut_ptr();
    let out_b_ptr = out_b.as_mut_ptr();
    let shift_idx = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    let zero = _mm256_setzero_ps();

    for oy in oy0..oy1 {
        let even_row = oy % 2 == 0;
        let ow_row = oy * ow;
        let (iy_a, iy_b) = if even_row {
            let iy = oy / 2;
            (iy, iy)
        } else {
            ((oy + 1) / 2, (oy - 1) / 2)
        };

        // Bottom edge: on the last odd row the ky=0 input row `(oy+1)/2` is
        // `h` and does not exist; use the bounds-checking scalar tail for it
        // instead of reading past the plane.
        if !even_row && iy_a >= h {
            convtranspose_row_tail(
                input,
                weight,
                in_batch_base,
                w_co_base,
                c_in,
                h,
                w,
                ow,
                in_plane,
                w_ci_stride,
                oy,
                0,
                false,
                out_a,
            );
            convtranspose_row_tail(
                input,
                weight,
                in_batch_base,
                w_co_base + 9,
                c_in,
                h,
                w,
                ow,
                in_plane,
                w_ci_stride,
                oy,
                0,
                false,
                out_b,
            );
            continue;
        }

        let mut ix0 = 0usize;
        while ix0 + 8 <= w && ix0 * 2 + 16 <= ow {
            let last = ix0 + 8 == w;
            let mut a_e = [zero; 2];
            let mut a_o1 = [zero; 2];
            let mut a_o2 = [zero; 2];
            let mut a_o3 = [zero; 2];
            let mut a_o4 = [zero; 2];

            for ci in 0..c_in {
                let in_a = in_ptr.add(in_batch_base + ci * in_plane + iy_a * w + ix0);
                let v0 = _mm256_loadu_ps(in_a);
                let v1 = if last {
                    _mm256_blend_ps(_mm256_permutevar8x32_ps(v0, shift_idx), zero, 0x80)
                } else {
                    _mm256_loadu_ps(in_a.add(1))
                };
                if even_row {
                    for ch in 0..2 {
                        let wb = w_ptr.add(w_co_base + ch * 9 + ci * w_ci_stride);
                        let w10 = _mm256_broadcast_ss(&*wb.add(3));
                        let w11 = _mm256_broadcast_ss(&*wb.add(4));
                        let w12 = _mm256_broadcast_ss(&*wb.add(5));
                        a_e[ch] = _mm256_fmadd_ps(v0, w11, a_e[ch]);
                        a_o1[ch] = _mm256_fmadd_ps(v1, w10, a_o1[ch]);
                        a_o2[ch] = _mm256_fmadd_ps(v0, w12, a_o2[ch]);
                    }
                } else {
                    let in_b = in_ptr.add(in_batch_base + ci * in_plane + iy_b * w + ix0);
                    let u0 = _mm256_loadu_ps(in_b);
                    let u1 = if last {
                        _mm256_blend_ps(_mm256_permutevar8x32_ps(u0, shift_idx), zero, 0x80)
                    } else {
                        _mm256_loadu_ps(in_b.add(1))
                    };
                    for ch in 0..2 {
                        let wb = w_ptr.add(w_co_base + ch * 9 + ci * w_ci_stride);
                        let w00 = _mm256_broadcast_ss(&*wb.add(0));
                        let w01 = _mm256_broadcast_ss(&*wb.add(1));
                        let w02 = _mm256_broadcast_ss(&*wb.add(2));
                        let w20 = _mm256_broadcast_ss(&*wb.add(6));
                        let w21 = _mm256_broadcast_ss(&*wb.add(7));
                        let w22 = _mm256_broadcast_ss(&*wb.add(8));
                        // `v` is the ky=0 row, `u` the ky=2 row.
                        a_e[ch] = _mm256_fmadd_ps(u0, w21, a_e[ch]);
                        a_e[ch] = _mm256_fmadd_ps(v0, w01, a_e[ch]);
                        a_o1[ch] = _mm256_fmadd_ps(u1, w20, a_o1[ch]);
                        a_o2[ch] = _mm256_fmadd_ps(u0, w22, a_o2[ch]);
                        a_o3[ch] = _mm256_fmadd_ps(v1, w00, a_o3[ch]);
                        a_o4[ch] = _mm256_fmadd_ps(v0, w02, a_o4[ch]);
                    }
                }
            }

            for ch in 0..2 {
                let acc_o = if even_row {
                    _mm256_add_ps(a_o1[ch], a_o2[ch])
                } else {
                    _mm256_add_ps(
                        _mm256_add_ps(a_o1[ch], a_o2[ch]),
                        _mm256_add_ps(a_o3[ch], a_o4[ch]),
                    )
                };
                let lo = _mm256_unpacklo_ps(a_e[ch], acc_o);
                let hi = _mm256_unpackhi_ps(a_e[ch], acc_o);
                let r0 = _mm256_permute2f128_ps(lo, hi, 0x20);
                let r1 = _mm256_permute2f128_ps(lo, hi, 0x31);
                let op = if ch == 0 { out_a_ptr } else { out_b_ptr };
                _mm256_storeu_ps(op.add(ow_row + ix0 * 2), r0);
                _mm256_storeu_ps(op.add(ow_row + ix0 * 2 + 8), r1);
            }

            ix0 += 8;
        }

        if ix0 * 2 < ow {
            convtranspose_row_tail(
                input,
                weight,
                in_batch_base,
                w_co_base,
                c_in,
                h,
                w,
                ow,
                in_plane,
                w_ci_stride,
                oy,
                ix0 * 2,
                even_row,
                out_a,
            );
            convtranspose_row_tail(
                input,
                weight,
                in_batch_base,
                w_co_base + 9,
                c_in,
                h,
                w,
                ow,
                in_plane,
                w_ci_stride,
                oy,
                ix0 * 2,
                even_row,
                out_b,
            );
        }
    }
}

/// Scalar tail for the last columns of one output row.
#[allow(clippy::too_many_arguments)]
fn convtranspose_row_tail(
    input: &[f32],
    weight: &[f32],
    in_batch_base: usize,
    w_co_base: usize,
    c_in: usize,
    h: usize,
    w: usize,
    ow: usize,
    in_plane: usize,
    w_ci_stride: usize,
    oy: usize,
    ox_start: usize,
    even_row: bool,
    out_slice: &mut [f32],
) {
    let py = if even_row { 1 } else { 0 };
    for ox in ox_start..ow {
        let px = if ox % 2 == 0 { 1 } else { 0 };
        let mut acc = 0.0_f32;
        for &(dy, dx) in &[(0_usize, 0_usize), (0, 2), (2, 0), (2, 2)] {
            let ky = py + dy;
            let kx = px + dx;
            if ky >= 3 || kx >= 3 {
                continue;
            }
            let iy_num = oy + 1;
            if iy_num < ky || (iy_num - ky) % 2 != 0 {
                continue;
            }
            let iy = (iy_num - ky) / 2;
            if iy >= h {
                continue;
            }
            let ix_num = ox + 1;
            if ix_num < kx || (ix_num - kx) % 2 != 0 {
                continue;
            }
            let ix = (ix_num - kx) / 2;
            if ix >= w {
                continue;
            }
            let in_base = in_batch_base + iy * w + ix;
            let w_base = w_co_base + ky * 3 + kx;
            for ci in 0..c_in {
                acc += input[in_base + ci * in_plane] * weight[w_base + ci * w_ci_stride];
            }
        }
        out_slice[oy * ow + ox] = acc;
    }
}

/// Scalar fallback for ConvTranspose s2k3 plane.
#[allow(clippy::too_many_arguments)]
fn convtranspose_s2k3_plane_scalar(
    input: &[f32],
    weight: &[f32],
    in_batch_base: usize,
    w_co_base: usize,
    c_in: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    in_plane: usize,
    w_ci_stride: usize,
    oy0: usize,
    oy1: usize,
    out_slice: &mut [f32],
) {
    let _ = oh;
    for oy in oy0..oy1 {
        let py = (oy + 1) % 2;
        for ox in 0..ow {
            let px = (ox + 1) % 2;
            let mut acc = 0.0_f32;
            for &(dy, dx) in &[(0_usize, 0_usize), (0, 2), (2, 0), (2, 2)] {
                let ky = py + dy;
                let kx = px + dx;
                if ky >= 3 || kx >= 3 {
                    continue;
                }
                let iy_num = oy + 1;
                if iy_num < ky {
                    continue;
                }
                let iy = (iy_num - ky) / 2;
                if iy >= h || (iy_num - ky) % 2 != 0 {
                    continue;
                }
                let ix_num = ox + 1;
                if ix_num < kx {
                    continue;
                }
                let ix = (ix_num - kx) / 2;
                if ix >= w || (ix_num - kx) % 2 != 0 {
                    continue;
                }
                let in_base = in_batch_base + iy * w + ix;
                let w_base = w_co_base + ky * 3 + kx;
                for ci in 0..c_in {
                    acc += input[in_base + ci * in_plane] * weight[w_base + ci * w_ci_stride];
                }
            }
            out_slice[oy * ow + ox] = acc;
        }
    }
}

/// Attempt to reformulate ConvTranspose2D as zero-pad + regular Conv2d.
///
/// Fallback for ConvTranspose configurations that don't match the direct s2k3
/// fast path but are still stride≥1, kernel≥1, dilation=1, group=1, symmetric
/// pads. Zero-pads the input with stride-1 zeros between elements, then runs
/// a regular Conv2d.
#[allow(clippy::too_many_arguments)]
fn try_convtranspose_as_conv(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) -> Option<Result<(), OnnxError>> {
    if input_shape.len() != 4
        || weight_shape.len() != 4
        || params.strides.len() != 2
        || params.dilations != [1, 1]
        || params.group != 1
        || params.pads.len() != 4
        || params.pads[0] != params.pads[2]
        || params.pads[1] != params.pads[3]
    {
        return None;
    }

    let (n, c_in, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let c_out = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let (s_h, s_w) = (params.strides[0], params.strides[1]);
    let (p_top, p_left) = (params.pads[0], params.pads[1]);

    let conv_pad_h = kh.checked_sub(p_top)?.checked_sub(1)?;
    let conv_pad_w = kw.checked_sub(p_left)?.checked_sub(1)?;

    let zp_h = (h - 1) * s_h + 1;
    let zp_w = (w - 1) * s_w + 1;

    let conv_oh = zp_h + 2 * conv_pad_h - kh + 1;
    let conv_ow = zp_w + 2 * conv_pad_w - kw + 1;

    let (oh, ow) = (out_shape[2], out_shape[3]);
    if oh > conv_oh || ow > conv_ow {
        return None;
    }

    // Build zero-padded input
    let zp_len = n * c_in * zp_h * zp_w;
    let mut zp_input = vec![0.0_f32; zp_len];
    for ni in 0..n {
        for ci in 0..c_in {
            let src_base = (ni * c_in + ci) * h * w;
            let dst_base = (ni * c_in + ci) * zp_h * zp_w;
            for iy in 0..h {
                for ix in 0..w {
                    zp_input[dst_base + iy * s_h * zp_w + ix * s_w] =
                        input[src_base + iy * w + ix];
                }
            }
        }
    }

    // Transpose weight from [C_in, C_out, kH, kW] to [C_out, C_in, kH, kW]
    // and flip the kernel spatially: ConvTranspose over a zero-interleaved
    // input equals a regular convolution with the spatially reversed kernel.
    let mut conv_weight = vec![0.0_f32; weight.len()];
    for ci in 0..c_in {
        for co in 0..c_out {
            let src_base = (ci * c_out + co) * kh * kw;
            let dst_base = (co * c_in + ci) * kh * kw;
            for ky in 0..kh {
                for kx in 0..kw {
                    conv_weight[dst_base + (kh - 1 - ky) * kw + (kw - 1 - kx)] =
                        weight[src_base + ky * kw + kx];
                }
            }
        }
    }

    let zp_tensor = oxionnx_core::Tensor {
        data: zp_input,
        shape: vec![n, c_in, zp_h, zp_w],
    };
    let w_tensor = oxionnx_core::Tensor {
        data: conv_weight,
        shape: vec![c_out, c_in, kh, kw],
    };

    // Stride 1: the upsampling is already encoded in the zero-interleaved
    // input (elements spaced `s` apart); the padding `k-1-p` completes the
    // equivalence.
    let conv_result = super::conv2d::conv2d(
        &zp_tensor,
        &w_tensor,
        None,
        [1, 1],
        [conv_pad_h, conv_pad_w, conv_pad_h, conv_pad_w],
        [1, 1],
        1,
    );

    let copy_h = oh.min(conv_oh);
    let copy_w = ow.min(conv_ow);
    for ni in 0..n {
        for co in 0..c_out {
            let src_base = (ni * c_out + co) * conv_oh * conv_ow;
            let dst_base = (ni * c_out + co) * oh * ow;
            for oy in 0..copy_h {
                for ox in 0..copy_w {
                    out[dst_base + oy * ow + ox] =
                        conv_result.data[src_base + oy * conv_ow + ox];
                }
            }
        }
    }

    Some(Ok(()))
}

/// Element count of a shape, rejecting an overflowing product.
fn volume(shape: &[usize], op: &str, what: &str) -> Result<usize, OnnxError> {
    shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: {what} shape {shape:?} overflows usize"))
        })
}

/// Rank-2 scatter-accumulate (the decoder / GAN shape).
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_rank2(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let (n, c_in, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let c_out_per_group = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let group = params.group;
    let c_out = c_out_per_group * group;
    let c_in_per_group = c_in / group;
    let oh = out_shape[2];
    let ow = out_shape[3];
    let [s_h, s_w] = [params.strides[0], params.strides[1]];
    let [d_h, d_w] = [params.dilations[0], params.dilations[1]];
    let [p_top, p_left] = [params.pads[0], params.pads[1]];

    for ni in 0..n {
        for g in 0..group {
            for ic in 0..c_in_per_group {
                let ci = g * c_in_per_group + ic;
                for iy in 0..h {
                    for ix in 0..w {
                        let in_val = input[((ni * c_in + ci) * h + iy) * w + ix];
                        for oc in 0..c_out_per_group {
                            let co = g * c_out_per_group + oc;
                            let w_base = (ci * c_out_per_group + oc) * kh * kw;
                            let o_base = (ni * c_out + co) * oh * ow;
                            for ky in 0..kh {
                                let oy_raw = iy * s_h + ky * d_h;
                                if oy_raw < p_top {
                                    continue;
                                }
                                let oy = oy_raw - p_top;
                                if oy >= oh {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ox_raw = ix * s_w + kx * d_w;
                                    if ox_raw < p_left {
                                        continue;
                                    }
                                    let ox = ox_raw - p_left;
                                    if ox >= ow {
                                        continue;
                                    }
                                    out[o_base + oy * ow + ox] +=
                                        in_val * weight[w_base + ky * kw + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Rank-generic scatter-accumulate.
///
/// Visits `(batch, group, in-channel, in-position, out-channel, kernel-offset)`
/// in exactly the order [`scatter_rank2`] does, so both produce bit-identical
/// sums. The leading spatial axes are resolved once per kernel row and only the
/// last axis is walked element-by-element.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_generic(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    params: &ConvTransposeParams<'_>,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let rank = input_shape.len() - 2;
    if rank == 0 {
        return;
    }
    let last = rank - 1;
    let n = input_shape[0];
    let c_in = input_shape[1];
    let in_spatial = &input_shape[2..];
    let kernel = &weight_shape[2..];
    let out_spatial = &out_shape[2..];
    let c_out_per_group = weight_shape[1];
    let group = params.group;
    let c_out = c_out_per_group * group;
    let c_in_per_group = c_in / group;

    let in_plane: usize = in_spatial.iter().product();
    let out_plane: usize = out_spatial.iter().product();
    let ksize: usize = kernel.iter().product();
    if in_plane == 0 || out_plane == 0 || ksize == 0 {
        return;
    }

    // Row-major strides of the output spatial axes.
    let mut out_stride = vec![1_usize; rank];
    for d in (0..last).rev() {
        out_stride[d] = out_stride[d + 1] * out_spatial[d + 1];
    }

    let k_last = kernel[last];
    let k_outer: usize = kernel[..last].iter().product();
    let stride_last = params.strides[last];
    let dilation_last = params.dilations[last];
    let pad_last = params.pads[last];
    let out_last = out_spatial[last];

    let mut iidx = vec![0_usize; rank];
    let mut kidx = vec![0_usize; last];

    for ni in 0..n {
        for g in 0..group {
            for ic in 0..c_in_per_group {
                let ci = g * c_in_per_group + ic;
                let in_base = (ni * c_in + ci) * in_plane;
                iidx.iter_mut().for_each(|v| *v = 0);
                for iflat in 0..in_plane {
                    let in_val = input[in_base + iflat];
                    for oc in 0..c_out_per_group {
                        let co = g * c_out_per_group + oc;
                        let w_base = (ci * c_out_per_group + oc) * ksize;
                        let o_base = (ni * c_out + co) * out_plane;
                        kidx.iter_mut().for_each(|v| *v = 0);
                        for ko in 0..k_outer {
                            let mut ok = true;
                            let mut off = 0_usize;
                            for d in 0..last {
                                let raw =
                                    iidx[d] * params.strides[d] + kidx[d] * params.dilations[d];
                                if raw < params.pads[d] {
                                    ok = false;
                                    break;
                                }
                                let o = raw - params.pads[d];
                                if o >= out_spatial[d] {
                                    ok = false;
                                    break;
                                }
                                off += o * out_stride[d];
                            }
                            if ok {
                                let w_row = w_base + ko * k_last;
                                let o_row = o_base + off;
                                for kl in 0..k_last {
                                    let raw = iidx[last] * stride_last + kl * dilation_last;
                                    if raw < pad_last {
                                        continue;
                                    }
                                    let o = raw - pad_last;
                                    if o >= out_last {
                                        continue;
                                    }
                                    out[o_row + o] += in_val * weight[w_row + kl];
                                }
                            }
                            if odometer_next(&mut kidx, &kernel[..last]) {
                                break;
                            }
                        }
                    }
                    if odometer_next(&mut iidx, in_spatial) {
                        break;
                    }
                }
            }
        }
    }
}

/// Write transposed conv2d result directly into a pre-allocated output buffer.
///
/// `out` must have length == product of `out_shape` elements.
/// `out` is zeroed before accumulation begins.
///
/// `pads`: `[top, left, bottom, right]`
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_into(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    // output_padding is not used here because the caller pre-computes out_shape
    // and passes it directly; the parameter is kept for API symmetry with the
    // public conv_transpose2d function.
    _output_padding: &[usize],
    dilations: &[usize],
    group: usize,
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<(), String> {
    if input.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: input must be 4D, got {}D",
            input.ndim()
        ));
    }
    if weight.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: weight must be 4D, got {}D",
            weight.ndim()
        ));
    }
    if out_shape.len() != 4 {
        return Err(format!(
            "conv_transpose2d: out_shape must be 4D, got rank {}",
            out_shape.len()
        ));
    }
    if strides.len() < 2 || pads.len() < 4 || dilations.len() < 2 {
        return Err(
            "conv_transpose2d: strides/dilations need 2 entries and pads needs 4".to_string(),
        );
    }
    conv_transpose_into(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        &ConvTransposeParams {
            strides: &strides[..2],
            pads: &pads[..4],
            dilations: &dilations[..2],
            group,
        },
        out,
        out_shape,
    )
    .map_err(|e| e.to_string())
}

/// ConvTranspose2D: fractionally-strided convolution (deconvolution)
/// input: [N, C_in, H, W]
/// weight: [C_in, C_out/group, kH, kW]
/// output: [N, C_out, oH, oW]
/// oH = stride*(H-1) + output_padding + ((kH-1)*dilation + 1) - pad_top - pad_bottom
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: [usize; 2],
    pads: [usize; 4], // [top, left, bottom, right]
    output_padding: [usize; 2],
    dilations: [usize; 2],
    group: usize,
) -> Result<Tensor, String> {
    // Validate dimensions early — before compute_conv_transpose2d_out_shape
    // indexes into the shape slices, which would panic on invalid inputs.
    if input.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: input must be 4D, got {}D",
            input.ndim()
        ));
    }
    if weight.ndim() != 4 {
        return Err(format!(
            "conv_transpose2d: weight must be 4D, got {}D",
            weight.ndim()
        ));
    }
    let out_shape = compute_conv_transpose2d_out_shape(
        &input.shape,
        &weight.shape,
        &strides,
        &pads,
        &output_padding,
        &dilations,
        group,
    )
    .map_err(|e| e.to_string())?;
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv_transpose2d_into(
        input,
        weight,
        bias,
        &strides,
        &pads,
        &output_padding,
        &dilations,
        group,
        &mut data,
        &out_shape,
    )?;
    Ok(Tensor::new(data, out_shape))
}

/// Rank-generic transposed convolution returning a fresh tensor.
///
/// `pads` uses the ONNX layout `[begin_0, …, end_{r-1}]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    strides: &[usize],
    pads: &[usize],
    output_padding: &[usize],
    dilations: &[usize],
    group: usize,
) -> Result<Tensor, OnnxError> {
    let out_shape = spatial::compute_conv_transpose_out_shape(
        "ConvTranspose",
        &input.shape,
        &weight.shape,
        strides,
        pads,
        output_padding,
        dilations,
        group,
    )?;
    let out_len: usize = out_shape.iter().product();
    let mut data = vec![0.0_f32; out_len];
    conv_transpose_into(
        &input.data,
        &input.shape,
        &weight.data,
        &weight.shape,
        bias.map(|b| b.data.as_slice()),
        &ConvTransposeParams {
            strides,
            pads,
            dilations,
            group,
        },
        &mut data,
        &out_shape,
    )?;
    Ok(Tensor::new(data, out_shape))
}
