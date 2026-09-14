// Modified by the GIMP LaMa inpainting fork of OxiONNX (2026-09): ADDS a block-swap transpose decomposition with AVX2 8x8 and 2-plane interleave/deinterleave kernels.
// See ../../../MODIFICATIONS.md for the full change list and rationale.

//! Basic shape manipulation operations: reshape, flatten, transpose, squeeze, unsqueeze.

use oxionnx_core::Tensor;

/// Normalize a possibly-negative ONNX axis against `rank`, returning a bounds-checked index.
///
/// Valid input range is `[-rank, rank-1]`, matching the axis convention shared by
/// Concat/Split/Slice/Squeeze/Gather/etc. Negative values count from the back
/// (`axis + rank`). Out-of-range values (including the classic huge-`usize` wraparound a raw
/// `(axis + rank as i64) as usize` cast produces for `axis < -rank`) are rejected instead of
/// silently indexing garbage.
pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(format!(
            "axis {axis} out of range for rank {rank} (expected [-{rank}, {}])",
            rank.saturating_sub(1)
        ));
    }
    Ok(a as usize)
}

/// Like [`normalize_axis`], but accepts the inclusive `[-rank, rank]` range that ONNX Flatten
/// uses (an axis equal to `rank` is legal and means "everything goes into the outer dim").
pub(crate) fn normalize_axis_inclusive(axis: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a > r {
        return Err(format!(
            "axis {axis} out of range for rank {rank} (expected [-{rank}, {rank}])"
        ));
    }
    Ok(a as usize)
}

/// Reshape `x` to the target `shape`, honouring the ONNX `allowzero` attribute.
///
/// Special entries in `shape`:
/// - `-1` infers a single dimension from the remaining element count (at most one allowed).
/// - `0` copies the corresponding input dimension when `allowzero` is `false` (the ONNX
///   default); when `allowzero` is `true` a `0` is a literal zero-size dimension (NumPy
///   semantics) and is taken verbatim.
///
/// When `allowzero` is `true`, combining a `-1` with an explicit `0` is ambiguous and
/// returns an error.
pub fn reshape(x: &Tensor, shape: &[i64], allowzero: bool) -> Result<Tensor, String> {
    let new_shape = resolve_reshape(&x.shape, x.numel(), shape, allowzero)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

/// Resolve a concrete output shape for Reshape from the target `shape` spec.
///
/// `input_dims` and `numel` describe the source tensor; `allowzero` selects whether a `0`
/// copies the input dimension (`false`, ONNX default) or is a literal zero (`true`).
pub fn resolve_reshape(
    input_dims: &[usize],
    numel: usize,
    shape: &[i64],
    allowzero: bool,
) -> Result<Vec<usize>, String> {
    let neg_count = shape.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 {
        return Err("reshape: at most one -1 allowed".into());
    }
    if let Some(&bad) = shape.iter().find(|&&d| d < -1) {
        return Err(format!(
            "reshape: shape entries must be >= -1 (the ONNX infer sentinel), got {bad}"
        ));
    }
    let has_explicit_zero = shape.contains(&0);
    if allowzero && neg_count == 1 && has_explicit_zero {
        return Err(
            "Reshape: cannot infer dimension (-1) when an explicit 0 is present under allowzero=1"
                .into(),
        );
    }

    // Resolve every non-(-1) entry first: a 0 either copies the input dim (allowzero=false)
    // or stays a literal zero (allowzero=true). The -1 entry is filled in afterwards. Entries
    // below -1 were already rejected above, so every remaining `d` here is >= 0.
    let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        let dim = if d == -1 {
            usize::MAX // placeholder, overwritten below
        } else if d == 0 && !allowzero {
            *input_dims.get(i).ok_or_else(|| {
                format!(
                    "reshape: 0 at index {i} has no matching input dimension (rank {})",
                    input_dims.len()
                )
            })?
        } else {
            d as usize
        };
        new_shape.push(dim);
    }

    if neg_count == 1 {
        let known: usize = new_shape.iter().filter(|&&d| d != usize::MAX).product();
        if known == 0 {
            return Err(format!(
                "reshape: cannot infer dimension (-1) when remaining dimensions multiply to 0 ({new_shape:?})"
            ));
        }
        let inferred = numel / known;
        for d in new_shape.iter_mut() {
            if *d == usize::MAX {
                *d = inferred;
            }
        }
    }

    if new_shape.iter().product::<usize>() != numel {
        return Err(format!(
            "reshape: element count mismatch ({numel} vs {new_shape:?})"
        ));
    }
    Ok(new_shape)
}

/// Resolve the `(outer, inner)` split for Flatten: `outer = prod(shape[..axis])`,
/// `inner = prod(shape[axis..])`. `axis` is normalized against the *inclusive* `[-r, r]`
/// range ONNX defines for Flatten (unlike the exclusive range most other axis-taking ops use).
///
/// Neither product is clamped to a minimum of 1: an empty leading/trailing slice legitimately
/// multiplies to 1 (the identity element), and a genuinely zero-size input dimension must stay
/// zero so the returned shape keeps `outer * inner == x.numel()`.
pub(crate) fn resolve_flatten_shape(
    x_shape: &[usize],
    axis: i64,
) -> Result<(usize, usize), String> {
    let ndim = x_shape.len();
    let ax = normalize_axis_inclusive(axis, ndim)?;
    let outer: usize = x_shape[..ax].iter().product();
    let inner: usize = x_shape[ax..].iter().product();
    Ok((outer, inner))
}

pub fn flatten(x: &Tensor, axis: i64) -> Result<Tensor, String> {
    let (outer, inner) = resolve_flatten_shape(&x.shape, axis)?;
    Ok(Tensor::new(x.data.clone(), vec![outer, inner]))
}

/// Transpose according to a permutation. If perm is empty, reverses all dims.
/// AVX2 8×8 register-blocked transpose of a `[p_len, q_len]` matrix.
///
/// This avoids the cache-set aliasing that makes the scalar tiled transpose
/// slow when the destination stride is a large power of two: `(1,192,64,64)`
/// with permutation `(3,0,1,2)` is a `[12288, 64]` transpose whose writes land
/// every 48 KB, so all 16 in-flight destination lines map to the same L1 set
/// and thrash. The 8×8 register block does 8 loads, 24 shuffles and 8 stores
/// per 64 elements with no in-flight line reuse — the destination rows are
/// written once each, in 32-byte runs.
///
/// # Safety
/// Caller must have confirmed AVX2. `src` readable for `p_len * q_len`,
/// `dst` writable for the same, and the two must not overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8_blocks(src: *const f32, dst: *mut f32, p_len: usize, q_len: usize) {
    // Tile traversal order matters: for `Q >> P` (the `(64,1,192,64)`
    // permutation is `[64, 12288] -> [12288, 64]`) walking p0 outermost
    // revisits every destination row once per p-tile — eight times for
    // P=64 — so each 64-byte destination line is fetched, half-written,
    // evicted and fetched again. Walking q0 outermost writes each short
    // destination row to completion (256 contiguous bytes) before moving on.
    // The scan order is irrelevant to the result.
    let q_outer = q_len > p_len;
    if q_outer {
        let mut q0 = 0usize;
        while q0 + 8 <= q_len {
            let mut p0 = 0usize;
            while p0 + 8 <= p_len {
                transpose_tile_8x8(src, dst, p0, q0, p_len, q_len);
                p0 += 8;
            }
            for p in p0..p_len {
                for q in q0..q0 + 8 {
                    *dst.add(q * p_len + p) = *src.add(p * q_len + q);
                }
            }
            q0 += 8;
        }
        for q in q0..q_len {
            for p in 0..p_len {
                *dst.add(q * p_len + p) = *src.add(p * q_len + q);
            }
        }
        return;
    }

    let mut p0 = 0usize;
    while p0 + 8 <= p_len {
        let mut q0 = 0usize;
        while q0 + 8 <= q_len {
            transpose_tile_8x8(src, dst, p0, q0, p_len, q_len);

            q0 += 8;
        }
        for q in q0..q_len {
            for p in p0..p0 + 8 {
                *dst.add(q * p_len + p) = *src.add(p * q_len + q);
            }
        }
        p0 += 8;
    }
    for p in p0..p_len {
        for q in 0..q_len {
            *dst.add(q * p_len + p) = *src.add(p * q_len + q);
        }
    }
}


/// One 8x8 transpose tile: load 8 rows of 8 floats from `src` (row stride
/// `q_len`), transpose in registers, store 8 rows of 8 into `dst` (row stride
/// `p_len`). Shared by both traversal orders of [`transpose_8x8_blocks`].
///
/// # Safety
/// Caller must have confirmed AVX2, and `p0 + 8 <= p_len`, `q0 + 8 <= q_len`,
/// with both buffers sized for their full `p_len * q_len` extent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_tile_8x8(
    src: *const f32,
    dst: *mut f32,
    p0: usize,
    q0: usize,
    p_len: usize,
    q_len: usize,
) {
    use core::arch::x86_64::*;
    let r0 = _mm256_loadu_ps(src.add(p0 * q_len + q0));
    let r1 = _mm256_loadu_ps(src.add((p0 + 1) * q_len + q0));
    let r2 = _mm256_loadu_ps(src.add((p0 + 2) * q_len + q0));
    let r3 = _mm256_loadu_ps(src.add((p0 + 3) * q_len + q0));
    let r4 = _mm256_loadu_ps(src.add((p0 + 4) * q_len + q0));
    let r5 = _mm256_loadu_ps(src.add((p0 + 5) * q_len + q0));
    let r6 = _mm256_loadu_ps(src.add((p0 + 6) * q_len + q0));
    let r7 = _mm256_loadu_ps(src.add((p0 + 7) * q_len + q0));

    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);

    let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    _mm256_storeu_ps(dst.add(q0 * p_len + p0), _mm256_permute2f128_ps(u0, u4, 0x20));
    _mm256_storeu_ps(dst.add((q0 + 1) * p_len + p0), _mm256_permute2f128_ps(u1, u5, 0x20));
    _mm256_storeu_ps(dst.add((q0 + 2) * p_len + p0), _mm256_permute2f128_ps(u2, u6, 0x20));
    _mm256_storeu_ps(dst.add((q0 + 3) * p_len + p0), _mm256_permute2f128_ps(u3, u7, 0x20));
    _mm256_storeu_ps(dst.add((q0 + 4) * p_len + p0), _mm256_permute2f128_ps(u0, u4, 0x31));
    _mm256_storeu_ps(dst.add((q0 + 5) * p_len + p0), _mm256_permute2f128_ps(u1, u5, 0x31));
    _mm256_storeu_ps(dst.add((q0 + 6) * p_len + p0), _mm256_permute2f128_ps(u2, u6, 0x31));
    _mm256_storeu_ps(dst.add((q0 + 7) * p_len + p0), _mm256_permute2f128_ps(u3, u7, 0x31));
}

/// Interleave two planes into pairs: `dst[2i] = a[i]`, `dst[2i+1] = b[i]`.
///
/// This is the `[2, Q] -> [Q, 2]` block swap — the FFC Fourier unit's
/// real/imaginary recombine, the transposed sibling of
/// [`deinterleave_2xn`]. The generic 8×8 path cannot express it (`P = 2` is
/// below one tile row), which left ~800 K elements per node to the scalar
/// tail; AVX2 processes 16 stored floats per iteration.
///
/// # Safety
/// Caller must have confirmed AVX2. `a` and `b` must each hold `len` floats,
/// `dst` must hold `2 * len`, and none of the three may overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave_2xn(a: *const f32, b: *const f32, dst: *mut f32, len: usize) {
    use core::arch::x86_64::*;
    let mut i = 0usize;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let lo = _mm256_unpacklo_ps(va, vb); // [a0 b0 a1 b1 | a4 b4 a5 b5]
        let hi = _mm256_unpackhi_ps(va, vb); // [a2 b2 a3 b3 | a6 b6 a7 b7]
        _mm256_storeu_ps(dst.add(i * 2), _mm256_permute2f128_ps(lo, hi, 0x20));
        _mm256_storeu_ps(dst.add(i * 2 + 8), _mm256_permute2f128_ps(lo, hi, 0x31));
        i += 8;
    }
    while i < len {
        *dst.add(i * 2) = *a.add(i);
        *dst.add(i * 2 + 1) = *b.add(i);
        i += 1;
    }
}

/// De-interleave `p_len` pairs: `even[i] = src[2i]`, `odd[i] = src[2i+1]`.
///
/// This is the `[P, 2] -> [2, P]` transpose — the FFC Fourier unit's
/// real/imaginary split, which the generic path walked one element at a time.
/// AVX2 processes 8 pairs (16 floats) per iteration: `vpermd` regroups two
/// input vectors into `[evens | odds]`, then `vperm2f128` produces the two
/// output vectors.
///
/// # Safety
/// Caller must have confirmed AVX2. `src` must be readable for `2 * p_len`
/// floats; `even` and `odd` writable for `p_len` floats each, and non-
/// overlapping with `src` and each other.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_2xn(src: *const f32, even: *mut f32, odd: *mut f32, p_len: usize) {
    use core::arch::x86_64::*;
    // Lane order: evens 0..4 then odds 0..4 within the vector produced by
    // `_mm256_permutevar8x32_ps` over `[a0 b0 a1 b1 a2 b2 a3 b3]`.
    let idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let mut i = 0usize;
    while i + 8 <= p_len {
        let v0 = _mm256_loadu_ps(src.add(i * 2));
        let v1 = _mm256_loadu_ps(src.add(i * 2 + 8));
        let p0 = _mm256_permutevar8x32_ps(v0, idx); // [a0 a1 a2 a3 | b0 b1 b2 b3]
        let p1 = _mm256_permutevar8x32_ps(v1, idx); // [a4 a5 a6 a7 | b4 b5 b6 b7]
        let evens = _mm256_permute2f128_ps(p0, p1, 0x20); // [a0..a3 | a4..a7]
        let odds = _mm256_permute2f128_ps(p0, p1, 0x31); // [b0..b3 | b4..b7]
        _mm256_storeu_ps(even.add(i), evens);
        _mm256_storeu_ps(odd.add(i), odds);
        i += 8;
    }
    while i < p_len {
        *even.add(i) = *src.add(i * 2);
        *odd.add(i) = *src.add(i * 2 + 1);
        i += 1;
    }
}

pub fn transpose(x: &Tensor, perm: &[usize]) -> Result<Tensor, String> {
    // A transpose is a permutation of the elements, so the output holds the
    // same count as the input and the buffer can be sized before the output
    // shape is known.
    let shape_numel: usize = x.shape.iter().product();
    let mut data = vec![0.0f32; shape_numel];
    let out_shape = transpose_into(x, perm, &mut data)?;
    Ok(Tensor::new(data, out_shape))
}

/// [`transpose`] writing into a caller-provided buffer.
///
/// `out` must hold at least `x.shape.iter().product()` elements and must not
/// overlap `x.data`. The slot path passes the node's pre-sized output buffer,
/// so the result is written once instead of being built in an intermediate
/// allocation and copied into the slot afterwards.
pub fn transpose_into(x: &Tensor, perm: &[usize], out: &mut [f32]) -> Result<Vec<usize>, String> {
    // Row-major (C-order) strides for `shape`: `strides[i] = prod(shape[i+1..])`.
    // Nested inside `transpose` (the only op owned in this file) rather than a
    // file-scope item shared with the rest of `basic.rs`.
    fn row_major_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0usize; shape.len()];
        let mut s = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = s;
            s *= shape[i];
        }
        strides
    }

    let ndim = x.ndim();
    let perm: Vec<usize> = if perm.is_empty() {
        (0..ndim).rev().collect()
    } else {
        perm.to_vec()
    };
    if perm.len() != ndim {
        return Err(format!(
            "transpose: perm len {} != ndim {}",
            perm.len(),
            ndim
        ));
    }
    // `perm` must be a genuine permutation of `0..ndim`: every entry in range, each used
    // exactly once. Without this, an out-of-range entry indexes `x.shape`/`in_strides` out of
    // bounds, and a repeated entry (e.g. perm=[0,0,1]) silently builds an `out_shape` whose
    // product no longer matches `x.numel()`.
    let mut seen = vec![false; ndim];
    for &p in &perm {
        if p >= ndim {
            return Err(format!(
                "transpose: perm entry {p} out of range for {ndim}D tensor"
            ));
        }
        if seen[p] {
            return Err(format!("transpose: perm entry {p} repeated"));
        }
        seen[p] = true;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    // Size the output from the *shape* product, not from `x.numel()`. Those differ: this
    // crate's `Tensor::numel()` returns `data.len()`, and `Tensor::new` only checks the two
    // agree via a `debug_assert!` (see its doc comment), so a release build can carry a
    // tensor whose buffer is longer than its shape describes -- and sizing the output from
    // that buffer would hand back an equally malformed result. `out_shape` is a permutation
    // of `x.shape`, so this is exactly `out_shape.iter().product()`. For a well-formed
    // tensor (the overwhelming common case) it is the same number as before.
    let shape_numel: usize = x.shape.iter().product();
    // A zero-size dimension anywhere makes `shape_numel == 0`: bail out before any of the
    // stride/division math below, which otherwise risks a `0 / 0` (the trailing-run length
    // can itself be 0 when a size-0 axis lands in the "run"). Nothing to move either way.
    if shape_numel == 0 {
        return Ok(out_shape);
    }
    // The opposite invariant violation -- a buffer *shorter* than the shape describes -- has
    // no correct transpose: some source element the output requires does not exist. Both the
    // tiled fast path and the general odometer below would index past the end of `x.data`, so
    // report it rather than panicking or silently zero-filling part of the result.
    if x.data.len() < shape_numel {
        return Err(format!(
            "transpose: data buffer holds {} elements but shape {:?} describes {shape_numel}",
            x.data.len(),
            x.shape
        ));
    }
    if out.len() < shape_numel {
        return Err(format!(
            "transpose: output buffer holds {} elements but the transpose needs {shape_numel}",
            out.len()
        ));
    }

    let in_strides = row_major_strides(&x.shape);

    // Fast path: every leading axis is untouched (`perm[i] == i`) and only the
    // trailing two axes are swapped -- e.g. plain 2D transpose (`[1,0]`), or
    // the attention `K^T` pattern at any batch rank (`[...,S,D] -> [...,D,S]`,
    // perm `[0,1,3,2]`). Each of the `batch = prod(shape[..ndim-2])` leading
    // combinations is then an independent 2D transpose of a contiguous
    // `[rows, cols]` plane.
    //
    // That inner 2D transpose is tiled into 16x16 blocks rather than walked as
    // one `rows`-times-`cols` double loop: measured against the untiled loop,
    // a plain double loop is a *regression* (down to ~0.7-0.9x the original
    // div/mod code) whenever `rows` lands on an unlucky power-of-two-ish
    // stride (512, 768, 2048 -- exactly the hidden sizes real transformer
    // shapes use), because every inner-loop write jumps `rows` elements and
    // aliases the same cache sets. Tiling keeps each block's scatter writes
    // confined to a few dozen cache lines; measured speedup with 16x16 tiles
    // ranged 1.0x-18x across a set of representative shapes (square, tall,
    // wide, power-of-two and not), with no observed regression versus the
    // original code.
    let is_trailing_swap = ndim >= 2
        && perm[..ndim - 2].iter().enumerate().all(|(i, &p)| p == i)
        && perm[ndim - 2] == ndim - 1
        && perm[ndim - 1] == ndim - 2;
    if is_trailing_swap {
        const BLOCK: usize = 16;
        let rows = x.shape[ndim - 2];
        let cols = x.shape[ndim - 1];
        let plane = rows * cols;
        let batch: usize = x.shape[..ndim - 2].iter().product();
        for b in 0..batch {
            let base = b * plane;
            let mut bi = 0;
            while bi < rows {
                let i_end = (bi + BLOCK).min(rows);
                let mut bj = 0;
                while bj < cols {
                    let j_end = (bj + BLOCK).min(cols);
                    for i in bi..i_end {
                        let row = &x.data[base + i * cols + bj..base + i * cols + j_end];
                        for (jj, &v) in row.iter().enumerate() {
                            out[base + (bj + jj) * rows + i] = v;
                        }
                    }
                    bj = j_end;
                }
                bi = i_end;
            }
        }
        return Ok(out_shape);
    }

    // ── Blocked-swap fast path ────────────────────────────────────────────
    //
    // Every permutation in the LaMa FFC graph is of the form
    //   perm = [identity prefix, Q-block, P-block, identity suffix]
    // i.e. a `[outer][P][Q][inner] -> [outer][Q][P][inner]` block swap:
    // (2,0,1,3), (1,2,0,3), (3,0,1,2), (1,2,3,0), (0,1,4,2,3), (0,1,3,4,2).
    // The generic odometer walks these one element at a time because their
    // contiguous trailing run is length 1 (or a tiny 2), which is why they
    // measured at 4-5 ms each against sub-millisecond 2D transposes.
    //
    // Decomposing them into `[outer][P][Q][inner]` lets the copy be either a
    // tiled 2D transpose (`inner == 1`) or a run copy of `inner` contiguous
    // floats, batched over `outer` -- both cache-friendly.
    {
        let mut o = 0usize;
        while o < ndim && perm[o] == o {
            o += 1;
        }
        let mut s = 0usize;
        while s < ndim - o && perm[ndim - 1 - s] == ndim - 1 - s {
            s += 1;
        }
        let mid = &perm[o..ndim - s];
        if !mid.is_empty() {
            // The Q block is the ascending run at the head of `mid`; the P
            // block is the rest. Find where the ascent breaks.
            let mut k = 1usize;
            while k < mid.len() && mid[k] == mid[k - 1] + 1 {
                k += 1;
            }
            let (q_axes, p_axes) = (k, mid.len() - k);
            if p_axes > 0 && q_axes > 0 {
                let q_ok = mid[..q_axes]
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == o + p_axes + i);
                let p_ok = mid[q_axes..]
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == o + i);
                if q_ok && p_ok {
                    let outer_len: usize = x.shape[..o].iter().product();
                    let p_len: usize = x.shape[o..o + p_axes].iter().product();
                    let q_len: usize = x.shape[o + p_axes..o + p_axes + q_axes]
                        .iter()
                        .product();
                    let inner_len: usize = x.shape[o + p_axes + q_axes..].iter().product();

                    const BLOCK: usize = 16;
                    // Special case: de-interleave a `[P, 2]` matrix into two
                    // contiguous rows (`Q == 2`, `inner == 1`) with AVX2.
                    #[cfg(target_arch = "x86_64")]
                    if q_len == 2 && inner_len == 1 && is_x86_feature_detected!("avx2") {
                        for ob in 0..outer_len {
                            let base = ob * p_len * 2;
                            // SAFETY: AVX2 confirmed above; `base` is in
                            // bounds of both tensors, which hold the same
                            // number of elements.
                            unsafe {
                                deinterleave_2xn(
                                    x.data.as_ptr().add(base),
                                    out.as_mut_ptr().add(base),
                                    out.as_mut_ptr().add(base + p_len),
                                    p_len,
                                );
                            }
                        }
                        return Ok(out_shape);
                    }

                    // Special case: interleave a `[2, Q]` matrix into `[Q, 2]`
                    // (`P == 2`, `inner == 1`) with AVX2.
                    #[cfg(target_arch = "x86_64")]
                    if p_len == 2 && inner_len == 1 && is_x86_feature_detected!("avx2") {
                        for ob in 0..outer_len {
                            let base = ob * p_len * q_len;
                            // SAFETY: AVX2 confirmed above; the two source
                            // planes and the destination are inside the
                            // (equal-length) buffers at `base`.
                            unsafe {
                                interleave_2xn(
                                    x.data.as_ptr().add(base),
                                    x.data.as_ptr().add(base + q_len),
                                    out.as_mut_ptr().add(base),
                                    q_len,
                                );
                            }
                        }
                        return Ok(out_shape);
                    }

                    // Inner-dim == 1: use the register-blocked AVX2 8x8
                    // transpose when available; a pure scalar 2D transpose
                    // otherwise. Larger inner dims fall through to the run
                    // copy below.
                    #[cfg(target_arch = "x86_64")]
                    if inner_len == 1 && is_x86_feature_detected!("avx2") {
                        for ob in 0..outer_len {
                            let base = ob * p_len * q_len;
                            // SAFETY: AVX2 confirmed; both operands span the
                            // same `p_len * q_len` elements at `base`.
                            unsafe {
                                transpose_8x8_blocks(
                                    x.data.as_ptr().add(base),
                                    out.as_mut_ptr().add(base),
                                    p_len,
                                    q_len,
                                );
                            }
                        }
                        return Ok(out_shape);
                    }

                    for ob in 0..outer_len {
                        let base = ob * p_len * q_len * inner_len;
                        let mut p0 = 0usize;
                        while p0 < p_len {
                            let p1 = (p0 + BLOCK).min(p_len);
                            let mut q0 = 0usize;
                            while q0 < q_len {
                                let q1 = (q0 + BLOCK).min(q_len);
                                for p in p0..p1 {
                                    for qq in q0..q1 {
                                        let src = base + (p * q_len + qq) * inner_len;
                                        let dst = base + (qq * p_len + p) * inner_len;
                                        out[dst..dst + inner_len]
                                            .copy_from_slice(&x.data[src..src + inner_len]);
                                    }
                                }
                                q0 = q1;
                            }
                            p0 = p1;
                        }
                    }
                    return Ok(out_shape);
                }
            }
        }
    }

    // General path: no division anywhere. First, find the longest suffix of
    // axes left untouched by `perm` (`perm[i] == i`) -- input and output store
    // that suffix with the same shape in the same order, so it is one
    // contiguous block in both tensors and can be moved with a single
    // `copy_from_slice` per outer step instead of walked element by element
    // (this is what turns the `[1,512,12,64] -> [1,12,512,64]` attention
    // permute, `perm = [0,2,1,3]`, into 6144 64-element `copy_from_slice`
    // calls instead of 393216 individually-divided element copies).
    let mut run = 0usize;
    while run < ndim && perm[ndim - 1 - run] == ndim - 1 - run {
        run += 1;
    }
    let run_len: usize = x.shape[ndim - run..].iter().product();
    let outer_ndim = ndim - run;
    if outer_ndim == 0 {
        // `perm` is the identity permutation (or `ndim == 0`): the whole
        // tensor is one contiguous run.
        out.copy_from_slice(&x.data);
        return Ok(out_shape);
    }

    // The remaining `outer_ndim` axes are walked with an increment-and-carry
    // odometer: `in_off`/`out_off` are updated by addition/subtraction only as
    // each coordinate ticks over, never recomputed from scratch by dividing
    // the flat index against a stride.
    let out_strides = row_major_strides(&out_shape);
    let mut coord = vec![0usize; outer_ndim];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    // The odometer's cycle length is the shape product -- driving it from `x.data.len()`
    // instead (as this did before) would run extra laps for an over-long buffer, re-walking
    // the same valid output range with wrapped source data and leaving a last-write-wins
    // result decided by the final lap rather than the first. This is the same invariant gap
    // `reduce_with_into`'s `.take(shape_numel)` and `shape_bounded_data`'s `.min(data.len())`
    // in `math/reduce.rs` were hardened against. `run_len` cannot be 0 here: it is a product
    // of a suffix of `x.shape`, so a 0 in it would have forced `shape_numel == 0` and taken
    // the early return above.
    let n_outer = shape_numel / run_len;
    for _ in 0..n_outer {
        out[out_off..out_off + run_len].copy_from_slice(&x.data[in_off..in_off + run_len]);
        for axis in (0..outer_ndim).rev() {
            coord[axis] += 1;
            in_off += in_strides[perm[axis]];
            out_off += out_strides[axis];
            if coord[axis] < out_shape[axis] {
                break;
            }
            in_off -= coord[axis] * in_strides[perm[axis]];
            out_off -= coord[axis] * out_strides[axis];
            coord[axis] = 0;
        }
    }
    Ok(out_shape)
}

/// Resolve the output shape for Squeeze: drop the given (possibly negative) axes, provided
/// each named axis is actually size-1; if `axes` is empty, drop every size-1 axis.
///
/// Squeezing away *every* axis yields the **empty** shape — a genuine rank-0 tensor — not the
/// rank-1 `[1]` this used to promote it to. ONNX distinguishes the two, and the difference is
/// observable one node later: `Shape` of a rank-0 tensor is a length-0 vector, so any
/// `Reshape`/`Concat` driven by it would otherwise be handed one dimension too many. NumPy
/// agrees: `np.squeeze(np.array([5.0])).shape == ()`.
pub(crate) fn resolve_squeeze_shape(x_shape: &[usize], axes: &[i64]) -> Result<Vec<usize>, String> {
    let ndim = x_shape.len();
    let resolved: Vec<usize> = if axes.is_empty() {
        (0..ndim).filter(|&i| x_shape[i] == 1).collect()
    } else {
        axes.iter()
            .map(|&a| normalize_axis(a, ndim))
            .collect::<Result<_, _>>()?
    };
    let new_shape: Vec<usize> = x_shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if resolved.contains(&i) && d == 1 {
                None
            } else {
                Some(d)
            }
        })
        .collect();
    Ok(new_shape)
}

/// Remove axes of size 1. If axes is empty, remove all size-1 dims.
pub fn squeeze(x: &Tensor, axes: &[i64]) -> Result<Tensor, String> {
    let new_shape = resolve_squeeze_shape(&x.shape, axes)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

/// Resolve the output shape for Unsqueeze: insert size-1 axes at the given (possibly negative)
/// positions. Per the ONNX spec, negative axes are normalized against the *output* rank
/// (`x_shape.len() + axes.len()`), not the shape as it grows one insertion at a time — using
/// the growing shape mis-normalizes negative axes and mis-orders mixed positive/negative lists.
pub(crate) fn resolve_unsqueeze_shape(
    x_shape: &[usize],
    axes: &[i64],
) -> Result<Vec<usize>, String> {
    let out_rank = x_shape.len() + axes.len();
    let mut normalized: Vec<usize> = axes
        .iter()
        .map(|&a| normalize_axis(a, out_rank))
        .collect::<Result<_, _>>()?;
    normalized.sort_unstable();
    for w in normalized.windows(2) {
        if w[0] == w[1] {
            return Err(format!(
                "unsqueeze: axis {} specified more than once (after normalization)",
                w[0]
            ));
        }
    }
    // Walk every output position once: an output index that is one of the (sorted, unique)
    // target axes gets a new size-1 dim; every other output position pulls the next input dim
    // in order. Since `normalized` is ascending, positions already emitted never shift.
    let mut new_shape = Vec::with_capacity(out_rank);
    let mut axes_iter = normalized.iter().peekable();
    let mut src = x_shape.iter();
    for pos in 0..out_rank {
        if axes_iter.peek() == Some(&&pos) {
            axes_iter.next();
            new_shape.push(1);
        } else {
            new_shape.push(
                *src.next().ok_or_else(|| {
                    "unsqueeze: internal error resolving output shape".to_string()
                })?,
            );
        }
    }
    Ok(new_shape)
}

/// Insert size-1 axes at given positions.
pub fn unsqueeze(x: &Tensor, axes: &[i64]) -> Result<Tensor, String> {
    let new_shape = resolve_unsqueeze_shape(&x.shape, axes)?;
    Ok(Tensor::new(x.data.clone(), new_shape))
}

#[cfg(test)]
mod axis_tests {
    use super::*;

    #[test]
    fn normalize_axis_accepts_full_range() {
        assert_eq!(normalize_axis(0, 3).unwrap(), 0);
        assert_eq!(normalize_axis(2, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-3, 3).unwrap(), 0);
    }

    #[test]
    fn normalize_axis_rejects_out_of_range() {
        assert!(normalize_axis(3, 3).is_err());
        assert!(normalize_axis(-4, 3).is_err());
        // Historically `(axis + rank as i64) as usize` on a deeply negative axis wrapped to a
        // huge usize instead of erroring; confirm that path is now a clean Err.
        assert!(normalize_axis(-10, 2).is_err());
        assert!(normalize_axis(i64::MIN, 2).is_err());
    }

    #[test]
    fn normalize_axis_inclusive_allows_rank() {
        assert_eq!(normalize_axis_inclusive(3, 3).unwrap(), 3);
        assert_eq!(normalize_axis_inclusive(-3, 3).unwrap(), 0);
        assert!(normalize_axis_inclusive(4, 3).is_err());
        assert!(normalize_axis_inclusive(-4, 3).is_err());
    }
}

/// [W2-perf-misc / a6-8] Correctness of the odometer/trailing-run rewrite of
/// `transpose`, which replaced a per-element chain of `ndim` hardware
/// divisions with (a) a tiled fast path for "swap the trailing two axes" and
/// (b) a trailing-contiguous-run `copy_from_slice` + increment/carry odometer
/// for everything else. `transpose` performs no floating-point arithmetic --
/// it only moves `f32` bit patterns around -- so every case below is checked
/// for *exact* equality, not a tolerance.
#[cfg(test)]
mod transpose_tests {
    use super::*;

    /// Independent reference implementation: the exact per-element div/mod
    /// decode `transpose` used before this change. Kept deliberately
    /// separate from the odometer/tiling logic under test (no shared helper,
    /// no shared code path) so a bug in the odometer's carry logic or in the
    /// trailing-swap tile bounds cannot also be present here and cancel out.
    fn reference_transpose(
        data: &[f32],
        shape: &[usize],
        perm: &[usize],
    ) -> (Vec<f32>, Vec<usize>) {
        let ndim = shape.len();
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let out_n: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_n];
        let mut in_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            in_strides[i] = s;
            s *= shape[i];
        }
        let mut out_strides = vec![0usize; ndim];
        let mut s = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = s;
            s *= out_shape[i];
        }
        for (out_idx, out_val) in out.iter_mut().enumerate() {
            let mut rem = out_idx;
            let mut in_idx = 0usize;
            for i in 0..ndim {
                let coord = rem / out_strides[i];
                rem %= out_strides[i];
                in_idx += coord * in_strides[perm[i]];
            }
            *out_val = data[in_idx];
        }
        (out, out_shape)
    }

    /// Every permutation of `0..shape.len()`, via Heap's algorithm.
    fn all_perms(n: usize) -> Vec<Vec<usize>> {
        let mut a: Vec<usize> = (0..n).collect();
        let mut out = vec![a.clone()];
        let mut c = vec![0usize; n];
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    a.swap(0, i);
                } else {
                    a.swap(c[i], i);
                }
                out.push(a.clone());
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        out
    }

    fn check(shape: &[usize], perm: &[usize]) {
        let n: usize = shape.iter().product();
        // Deterministic, non-trivial values (not just the index itself, so an
        // accidental identity-like bug can't hide behind `data[i] == i`).
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.75 - 3.25).collect();
        let x = Tensor::new(data.clone(), shape.to_vec());
        let got = transpose(&x, perm)
            .unwrap_or_else(|e| panic!("transpose{shape:?} perm={perm:?} failed: {e}"));
        let (want_data, want_shape) = reference_transpose(&data, shape, perm);
        assert_eq!(
            got.shape, want_shape,
            "shape mismatch for {shape:?} perm={perm:?}"
        );
        assert_eq!(
            got.data, want_data,
            "data mismatch (must be bit-identical -- transpose moves floats, it doesn't compute them) for {shape:?} perm={perm:?}"
        );
    }

    #[test]
    fn matches_numpy_reference_general_odometer_path() {
        // perm = [0, 2, 1, 3]: last axis stays put (the trailing-run the
        // general path exploits) but axes 1/2 are not the trailing two, so
        // this does NOT take the trailing-swap fast path -- exactly the
        // `[1,512,12,64] -> [1,12,512,64]` attention-permute shape from a6-8,
        // shrunk to a size small enough to hand-verify.
        //
        // ```python
        // x = (np.arange(2*3*2*4, dtype=np.float32) * 0.5 - 3.0).reshape(2,3,2,4)
        // np.transpose(x, (0,2,1,3)).flatten()
        // ```
        let shape = [2usize, 3, 2, 4];
        let perm = [0usize, 2, 1, 3];
        let x_data: Vec<f32> = (0..48).map(|i| i as f32 * 0.5 - 3.0).collect();
        let x = Tensor::new(x_data, shape.to_vec());
        let y = transpose(&x, &perm).expect("transpose failed");
        assert_eq!(y.shape, vec![2, 2, 3, 4]);
        let expected: Vec<f32> = vec![
            -3.0, -2.5, -2.0, -1.5, 1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5, -1.0, -0.5, 0.0, 0.5,
            3.0, 3.5, 4.0, 4.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 13.0, 13.5, 14.0, 14.5,
            17.0, 17.5, 18.0, 18.5, 11.0, 11.5, 12.0, 12.5, 15.0, 15.5, 16.0, 16.5, 19.0, 19.5,
            20.0, 20.5,
        ];
        assert_eq!(y.data, expected);
    }

    #[test]
    fn matches_numpy_reference_2d_full_swap() {
        // ```python
        // x = (np.arange(15, dtype=np.float32) * 0.25 - 1.0).reshape(3,5)
        // np.transpose(x, (1,0)).flatten()
        // ```
        let x_data: Vec<f32> = (0..15).map(|i| i as f32 * 0.25 - 1.0).collect();
        let x = Tensor::new(x_data, vec![3, 5]);
        let y = transpose(&x, &[1, 0]).expect("transpose failed");
        assert_eq!(y.shape, vec![5, 3]);
        let expected: Vec<f32> = vec![
            -1.0, 0.25, 1.5, -0.75, 0.5, 1.75, -0.5, 0.75, 2.0, -0.25, 1.0, 2.25, 0.0, 1.25, 2.5,
        ];
        assert_eq!(y.data, expected);
    }

    #[test]
    fn trailing_swap_fast_path_matches_reference_across_tile_boundaries() {
        // rows/cols straddle the 16-element tile edge in both directions
        // (smaller than one tile, exactly one tile, and multiple tiles with a
        // ragged remainder), and `batch > 1` so the per-batch `plane` offset
        // is exercised too.
        for &(batch, rows, cols) in &[
            (1usize, 3usize, 5usize),
            (1, 16, 16),
            (2, 17, 15),
            (3, 31, 33),
            (4, 40, 8),
        ] {
            check(&[batch, rows, cols], &[0, 2, 1]);
            // Rank 4: leading batch axis untouched, trailing two swapped --
            // the `[B,H,S,D] -> [B,H,D,S]` attention K^T pattern.
            check(&[2, batch, rows, cols], &[0, 1, 3, 2]);
        }
    }

    #[test]
    fn general_path_matches_reference_for_every_5d_permutation() {
        // Small enough to be fast (5! = 120 permutations, ~40 elements each)
        // but big enough to exercise every trailing-run length from 0..5 and
        // both fast-path/general-path dispatch.
        let shape = [2usize, 1, 3, 2, 1];
        for perm in all_perms(5) {
            check(&shape, &perm);
        }
    }

    #[test]
    fn zero_size_dimension_does_not_panic() {
        // A dynamic-shape model can legitimately resolve a dimension to 0.
        // Division by the trailing-run length (`out_n / run_len`) must not
        // fire when `out_n == 0`, regardless of which axis carries the 0.
        for shape in [[0usize, 4, 3], [4, 0, 3], [4, 3, 0]] {
            let x = Tensor::new(Vec::new(), shape.to_vec());
            let y = transpose(&x, &[2, 0, 1]).expect("transpose of an empty tensor must not panic");
            assert_eq!(y.numel(), 0);
            assert_eq!(y.shape, vec![shape[2], shape[0], shape[1]]);
        }
    }

    #[test]
    fn identity_permutation_is_a_pure_copy() {
        for shape in [vec![5usize], vec![2, 3], vec![2, 3, 4], vec![1], vec![]] {
            // A 0-d shape's element count is the empty product, 1 -- `Tensor`
            // treats a 0-d tensor as a single scalar element.
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let x = Tensor::new(data.clone(), shape.clone());
            let perm: Vec<usize> = (0..shape.len()).collect();
            let y = transpose(&x, &perm).expect("identity transpose must not fail");
            assert_eq!(y.shape, shape);
            assert_eq!(y.data, data);
        }
    }

    #[test]
    fn empty_perm_reverses_all_axes_like_before() {
        let x = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
        let y = transpose(&x, &[]).expect("transpose with empty perm must not fail");
        assert_eq!(y.shape, vec![4, 3, 2]);
        let (want_data, want_shape) = reference_transpose(&x.data, &x.shape, &[2, 1, 0]);
        assert_eq!(y.shape, want_shape);
        assert_eq!(y.data, want_data);
    }

    #[test]
    fn single_element_tensor() {
        let x = Tensor::new(vec![42.0], vec![1, 1, 1]);
        let y = transpose(&x, &[2, 0, 1]).expect("transpose failed");
        assert_eq!(y.shape, vec![1, 1, 1]);
        assert_eq!(y.data, vec![42.0]);
    }

    #[test]
    fn invalid_perm_still_rejected() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(
            transpose(&x, &[0, 0]).is_err(),
            "repeated perm entry must error"
        );
        assert!(
            transpose(&x, &[0, 2]).is_err(),
            "out-of-range perm entry must error"
        );
        assert!(transpose(&x, &[0]).is_err(), "perm len != ndim must error");
    }
}
