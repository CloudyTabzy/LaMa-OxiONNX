//! LaMa inpainting sidecar worker — pure-Rust OxiONNX backend.
//!
//! This is a drop-in replacement for the ORT-based `lama-worker` that uses
//! OxiONNX (pure Rust ONNX inference, zero C/C++ deps) instead of
//! `ort` (ONNX Runtime C++ bindings). The CLI, file I/O, image processing
//! pipeline, and output format are identical to the ORT worker so the GIMP
//! plug-in can swap the binary without any changes.
//!
//! Boundary markers `[LAMA_MARKER] phase <name>` are emitted on stderr
//! with `flush` between validate / inference_start / inference_done /
//! result_written so the parent GIMP process can parse phase transitions.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use ndarray::{s, Array2, Array3, Array4};
use oxionnx::{OptLevel, Session, Tensor};

/// Mask threshold in 0..=255. Any nonzero pixel becomes "inpaint",
/// matching the reference LaMa behavior (predict.py: `(mask > 0) * 1`).
const MASK_THRESHOLD: u8 = 0;

#[derive(Parser, Debug)]
#[command(
    name = "lama-worker-oxionnx",
    version,
    about = "LaMa inpainting sidecar worker (pure Rust, OxiONNX backend)"
)]
struct Args {
    /// Path to the input PNG.
    #[arg(long)]
    image: PathBuf,
    /// Grayscale mask PNG path.
    #[arg(long)]
    mask: PathBuf,
    /// Path to the result PNG that the worker writes (RGBA).
    #[arg(long)]
    output: PathBuf,
    /// Path to the ONNX model file.
    #[arg(long)]
    model: PathBuf,
}

fn main() {
    if let Err(err) = run() {
        let mut message = format!("{:#}", err);
        message = message.split_whitespace().collect::<Vec<_>>().join(" ");
        if message.is_empty() {
            message = err.to_string();
        }
        if message.len() > 500 {
            message.truncate(497);
            message.push_str("...");
        }
        eprintln!("ERROR: {}", message);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    marker("validate");
    run_inpaint(&args)
}

fn run_inpaint(args: &Args) -> Result<()> {
    let image_path = require_file(&args.image, "image")?;
    let mask_path = require_file(&args.mask, "mask")?;
    let model_path = require_file(&args.model, "model")?;
    let output_path = output_path(&args.output)?;

    // Load the input as RGBA so we can preserve the exact 8-bit alpha bytes.
    let img_rgba_u8 = image::open(&image_path)
        .with_context(|| format!("failed to open image: {}", image_path.display()))?
        .to_rgba8();
    let (width, height) = (img_rgba_u8.width() as usize, img_rgba_u8.height() as usize);
    if width == 0 || height == 0 {
        bail!("image dimensions must be nonzero");
    }

    // Load the mask as a single-channel u8 grayscale.
    let mask_gray_u8 = image::open(&mask_path)
        .with_context(|| format!("failed to open mask: {}", mask_path.display()))?
        .to_luma8();
    let (mask_w, mask_h) = (mask_gray_u8.width() as usize, mask_gray_u8.height() as usize);
    if mask_w != width || mask_h != height {
        bail!(
            "image/mask dimensions differ: {}x{} vs {}x{}",
            width,
            height,
            mask_w,
            mask_h
        );
    }

    // Build the f32 HWC image, the bool HW mask (model input), and the
    // soft f32 HW mask (compositing).
    let image_hwc: Array3<f32> = {
        let mut arr = Array3::<f32>::zeros((height, width, 3));
        for y in 0..height {
            for x in 0..width {
                let p = img_rgba_u8[(x as u32, y as u32)];
                arr[[y, x, 0]] = p[0] as f32 / 255.0;
                arr[[y, x, 1]] = p[1] as f32 / 255.0;
                arr[[y, x, 2]] = p[2] as f32 / 255.0;
            }
        }
        arr
    };
    let mask_soft: Array2<f32> = {
        let mut arr = Array2::<f32>::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                arr[[y, x]] = mask_gray_u8[(x as u32, y as u32)].0[0] as f32 / 255.0;
            }
        }
        arr
    };
    let mask: Array2<bool> = {
        let mut arr = Array2::<bool>::from_elem((height, width), false);
        for y in 0..height {
            for x in 0..width {
                arr[[y, x]] = mask_gray_u8[(x as u32, y as u32)].0[0] > MASK_THRESHOLD;
            }
        }
        arr
    };

    marker("inference_start");
    let inference_start = Instant::now();

    // Full-image fast path: pad to mod-16, single inference.
    let (ph16, pw16, pt, pl) = pad16_dims(height, width);
    let img_pad = reflect_pad_3d(&image_hwc, pt, pl, ph16 - height - pt, pw16 - width - pl);
    let mask_pad = reflect_pad_2d(&mask, pt, pl, ph16 - height - pt, pw16 - width - pl);

    // Load the ONNX model and run inference.
    // Enable profiling when OXIONNX_PROFILE=1 is set.
    let profile = std::env::var("OXIONNX_PROFILE").ok().as_deref()
        .map_or(false, |v| !matches!(v, "" | "0" | "false" | "no" | "off"));

    // Session cache: parsing the 198 MB ONNX protobuf costs several hundred
    // milliseconds per run. `save_optimized`/`load_optimized` persist the
    // already-optimized graph + weights, skipping protobuf decode and every
    // optimization pass.
    //
    // Validity has two independent parts:
    //   * the **model** must not have changed (size + mtime, below);
    //   * the **binary's optimizer** must not have changed — embedded in the
    //     file name via `SESSION_CACHE_REVISION`, because a cache saved by an
    //     older build holds the older fused graph. Without the revision, an
    //     in-place binary upgrade silently kept running the pre-upgrade graph
    //     (we hit exactly this when the fusion passes landed), which is a
    //     correctness-shaped failure, not merely a slow one.
    let use_cache = std::env::var("OXIONNX_NO_SESSION_CACHE").ok().as_deref()
        .map_or(true, |v| matches!(v, "" | "0" | "false" | "no" | "off"));

    // QA knob: force a graph-optimization level (`OXIONNX_OPT_LEVEL` =
    // none|basic|extended|all) to bisect optimizer-related numerics. Pair it
    // with `OXIONNX_NO_SESSION_CACHE=1`, otherwise the cached graph wins.
    let opt_level = match std::env::var("OXIONNX_OPT_LEVEL").ok().as_deref() {
        Some("none") => Some(OptLevel::None),
        Some("basic") => Some(OptLevel::Basic),
        Some("extended") => Some(OptLevel::Extended),
        Some("all") => Some(OptLevel::All),
        _ => None,
    };

    let cache_path = session_cache_path(&model_path);

    // Best-effort: drop the pre-revision cache name from earlier builds so a
    // 200 MB dead file does not sit next to the model forever.
    let legacy_path = model_path.with_extension("onnx.oxicache");
    if legacy_path != cache_path {
        let _ = std::fs::remove_file(&legacy_path);
    }

    let cache_is_fresh = || -> bool {
        let Ok(model_meta) = std::fs::metadata(&model_path) else {
            return false;
        };
        let Ok(cache_meta) = std::fs::metadata(&cache_path) else {
            return false;
        };
        let mtime = |m: &std::fs::Metadata| {
            m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        };
        cache_meta.len() > 0 && mtime(&cache_meta) >= mtime(&model_meta)
    };

    let session = if let Some(level) = opt_level {
        Session::builder()
            .with_optimization_level(level)
            .load(&model_path)
            .with_context(|| format!("failed to load ONNX model: {}", model_path.display()))?
    } else if profile {
        Session::builder()
            .with_profiling()
            .load(&model_path)
            .with_context(|| format!("failed to load ONNX model: {}", model_path.display()))?
    } else if use_cache && cache_is_fresh() {
        match Session::load_optimized(&cache_path) {
            Ok(s) => s,
            Err(_) => {
                let s = Session::from_file(&model_path).with_context(|| {
                    format!("failed to load ONNX model: {}", model_path.display())
                })?;
                save_cache_atomic(&s, &cache_path);
                s
            }
        }
    } else {
        let s = Session::from_file(&model_path).with_context(|| {
            format!("failed to load ONNX model: {}", model_path.display())
        })?;
        if use_cache {
            save_cache_atomic(&s, &cache_path);
        }
        s
    };

    let input_names = session.input_names();
    if input_names.len() != 2 {
        bail!("expected exactly 2 model inputs, got {}", input_names.len());
    }
    let output_names = session.output_names();
    let output_name = output_names
        .first()
        .ok_or_else(|| anyhow!("model has no outputs"))?;

    // Convert HWC to CHW for the model.
    let img_chw: Array4<f32> = {
        let mut arr = Array4::<f32>::zeros((1, 3, ph16, pw16));
        for y in 0..ph16 {
            for x in 0..pw16 {
                for ch in 0..3 {
                    arr[[0, ch, y, x]] = img_pad[[y, x, ch]];
                }
            }
        }
        arr
    };
    let mask_chw: Array4<f32> = {
        let mut arr = Array4::<f32>::zeros((1, 1, ph16, pw16));
        for y in 0..ph16 {
            for x in 0..pw16 {
                arr[[0, 0, y, x]] = if mask_pad[[y, x]] { 1.0_f32 } else { 0.0_f32 };
            }
        }
        arr
    };

    // Build input tensors.
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert(
        input_names[0].as_str(),
        Tensor::new(img_chw.into_raw_vec_and_offset().0, vec![1, 3, ph16, pw16]),
    );
    inputs.insert(
        input_names[1].as_str(),
        Tensor::new(mask_chw.into_raw_vec_and_offset().0, vec![1, 1, ph16, pw16]),
    );

    let load_elapsed = inference_start.elapsed();
    let run_start = Instant::now();

    // Run inference.
    let outputs = session
        .run(&inputs)
        .with_context(|| "OxiONNX inference failed")?;

    // QA knob: dump every graph output as raw f32 + a `.shape` file. Used
    // with a tap-augmented model to diff per-node tensors against a
    // reference engine (`LAMA_OXIONNX_DUMP_TAPS=<dir>`).
    if let Ok(dump_dir) = std::env::var("LAMA_OXIONNX_DUMP_TAPS") {
        let dump_dir = PathBuf::from(dump_dir);
        if std::fs::create_dir_all(&dump_dir).is_ok() {
            for (name, tensor) in &outputs {
                let safe: String = name
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let shape = tensor
                    .shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                let _ = std::fs::write(dump_dir.join(format!("{safe}.shape")), shape);
                let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
                for v in &tensor.data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let _ = std::fs::write(dump_dir.join(format!("{safe}.bin")), bytes);
            }
        }
    }

    let run_elapsed = run_start.elapsed();
    eprintln!(
        "[LAMA_MARKER] timing load_ms={} run_ms={}",
        load_elapsed.as_millis(),
        run_elapsed.as_millis()
    );

    // Extract output.
    let output_tensor = outputs
        .get(output_name.as_str())
        .ok_or_else(|| anyhow!("model output `{}` missing", output_name))?;

    // Convert CHW output back to HWC.
    let out_shape = &output_tensor.shape;
    if out_shape.len() != 4 {
        bail!("expected 4D model output, got {}D", out_shape.len());
    }
    let (_, _, out_h, out_w) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    // Convert CHW output back to HWC. The LaMa model outputs 0..255; the
    // rest of the pipeline (composite, write) works in 0..1 like the
    // reference Python/ORT workers, which divide by 255 right here.
    let out_hwc: Array3<f32> = {
        let inv255 = 1.0f32 / 255.0;
        let mut arr = Array3::<f32>::zeros((out_h, out_w, 3));
        let data = &output_tensor.data;
        for y in 0..out_h {
            for x in 0..out_w {
                for ch in 0..3 {
                    arr[[y, x, ch]] = data[ch * out_h * out_w + y * out_w + x] * inv255;
                }
            }
        }
        arr
    };

    let inference_elapsed = inference_start.elapsed();
    eprintln!(
        "[LAMA_MARKER] inference_done elapsed_ms={}",
        inference_elapsed.as_millis()
    );

    // Dump per-node profiling data if enabled.
    if profile {
        if let Some(results) = session.profiling_results() {
            // Dump ALL nodes to a CSV for offline analysis.
            {
                use std::io::Write as _;
                if let Ok(mut f) = std::fs::File::create("test_data/profile_all_nodes.csv") {
                    let _ = writeln!(f, "ms,op_type,node_name");
                    for p in results.iter() {
                        let _ = writeln!(
                            f,
                            "{:.4},{},{}",
                            p.duration.as_secs_f64() * 1000.0,
                            p.op_type,
                            p.node_name
                        );
                    }
                }
            }
            eprintln!("\n=== OxiONNX per-node profiling (top 50 by time) ===");
            let mut sorted: Vec<_> = results.iter().collect();
            sorted.sort_by(|a, b| b.duration.cmp(&a.duration));
            for (i, p) in sorted.iter().take(50).enumerate() {
                eprintln!(
                    "  {:3}. {:20} {:30} {:8.3} ms",
                    i + 1,
                    p.op_type,
                    p.node_name.chars().take(30).collect::<String>(),
                    p.duration.as_secs_f64() * 1000.0
                );
            }
            // Also dump totals by op type
            use std::collections::HashMap;
            let mut by_op: HashMap<&str, f64> = HashMap::new();
            let mut count_by_op: HashMap<&str, usize> = HashMap::new();
            for p in &results {
                *by_op.entry(p.op_type.as_str()).or_default() += p.duration.as_secs_f64() * 1000.0;
                *count_by_op.entry(p.op_type.as_str()).or_default() += 1;
            }
            let mut op_sorted: Vec<_> = by_op.iter().collect();
            op_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            eprintln!("\n=== Total time by op type ===");
            for (op, ms) in op_sorted.iter().take(20) {
                let count = count_by_op.get(*op).unwrap_or(&0);
                eprintln!("  {:20} {:6} nodes  {:10.1} ms", op, count, ms);
            }
            let total_ms: f64 = by_op.values().sum();
            eprintln!("  {:20} {:6}        {:10.1} ms", "TOTAL", results.len(), total_ms);
        }
    }

    // Crop back to original size and soft-composite.
    let out = out_hwc
        .slice(s![pt..pt + height, pl..pl + width, ..])
        .to_owned();
    let result = soft_composite(&image_hwc, &out, &mask_soft);

    // Write output PNG, preserving the original alpha channel.
    write_output_rgba(&result, &img_rgba_u8, width, height, &output_path)?;

    marker("result_written");
    Ok(())
}

// ---------------------------------------------------------------------------
// Session-cache helpers
// ---------------------------------------------------------------------------

/// Bump this whenever the vendored optimizer's graph rewriting changes — any
/// fusion pass, any change to what `save_optimized` stores. It is embedded in
/// the cache file name (`<model>.r<N>.oxicache`), so a cache written by an
/// older build is never loaded by a newer binary: a stale cache would replay
/// the *old* fused graph, which is a correctness problem dressed as a
/// performance one.
///
/// History: r1 — initial (fuse_add_batchnorm + ConvTranspose BatchNorm fold).
const SESSION_CACHE_REVISION: u32 = 1;

/// `<model.onnx>` → `<model.onnx>.r{N}.oxicache`, in the model's directory.
fn session_cache_path(model_path: &Path) -> PathBuf {
    let file = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.onnx");
    model_path.with_file_name(format!("{file}.r{SESSION_CACHE_REVISION}.oxicache"))
}

/// Persist the optimized session without ever leaving a half-written cache in
/// place: the bytes go to a sibling temp file first and only replace the real
/// cache once `save_optimized` has succeeded. A crash, a full disk, or two
/// workers racing leaves the previous cache (or none) — both of which the
/// loader handles — rather than a truncated file.
fn save_cache_atomic(session: &Session, path: &Path) {
    let tmp = path.with_extension("tmp");
    if session.save_optimized(&tmp).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    // `rename` does not replace an existing file on Windows; remove first.
    let _ = std::fs::remove_file(path);
    if std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

fn require_file(path: &Path, label: &str) -> Result<PathBuf> {
    if path.exists() {
        Ok(path.to_path_buf())
    } else {
        Err(anyhow!("{} file not found: {}", label, path.display()))
    }
}

fn output_path(path: &Path) -> Result<PathBuf> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create output dir: {}", parent.display()))?;
        }
    }
    Ok(path.to_path_buf())
}

/// Compute the mod-16 padded dimensions and the top/left pad amounts.
/// Matches the Python `_pad16_dims`.
fn pad16_dims(h: usize, w: usize) -> (usize, usize, usize, usize) {
    let ph = h.div_ceil(16) * 16;
    let pw = w.div_ceil(16) * 16;
    let pt = (ph - h) / 2;
    let pl = (pw - w) / 2;
    (ph, pw, pt, pl)
}

/// Reflect-pad a 3D HWC array on all four sides.
fn reflect_pad_3d(
    arr: &Array3<f32>,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Array3<f32> {
    let (h, w, c) = (arr.dim().0, arr.dim().1, arr.dim().2);
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let mut out = Array3::<f32>::zeros((new_h, new_w, c));
    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = reflect_index(y as i64, pad_top as i64, h as i64);
            let src_x = reflect_index(x as i64, pad_left as i64, w as i64);
            for ch in 0..c {
                out[[y, x, ch]] = arr[[src_y as usize, src_x as usize, ch]];
            }
        }
    }
    out
}

/// Reflect-pad a 2D HW array on all four sides.
fn reflect_pad_2d(
    arr: &Array2<bool>,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Array2<bool> {
    let (h, w) = (arr.dim().0, arr.dim().1);
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let mut out = Array2::<bool>::from_elem((new_h, new_w), false);
    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = reflect_index(y as i64, pad_top as i64, h as i64);
            let src_x = reflect_index(x as i64, pad_left as i64, w as i64);
            out[[y, x]] = arr[[src_y as usize, src_x as usize]];
        }
    }
    out
}

/// Map a padded index back to the source array using reflection.
/// Mirrors numpy's `mode="reflect"` (which does NOT include the edge pixel).
fn reflect_index(i: i64, pad_before: i64, size: i64) -> i64 {
    let idx = i - pad_before;
    if idx < 0 {
        -idx
    } else if idx >= size {
        2 * (size - 1) - idx
    } else {
        idx
    }
}

/// Blend the model output with the original image using the soft mask.
/// `result = soft_mask * out + (1 - soft_mask) * orig`
fn soft_composite(
    orig: &Array3<f32>,
    out: &Array3<f32>,
    mask_soft: &Array2<f32>,
) -> Array3<f32> {
    let (h, w, c) = (orig.dim().0, orig.dim().1, orig.dim().2);
    let mut result = Array3::<f32>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            let a = mask_soft[[y, x]];
            for ch in 0..c {
                result[[y, x, ch]] = a * out[[y, x, ch]] + (1.0 - a) * orig[[y, x, ch]];
            }
        }
    }
    result
}

/// Write the result as an RGBA PNG, preserving the original alpha channel.
fn write_output_rgba(
    result: &Array3<f32>,
    orig_rgba: &image::RgbaImage,
    width: usize,
    height: usize,
    output_path: &Path,
) -> Result<()> {
    let mut img_buf = image::RgbaImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let orig_px = orig_rgba[(x as u32, y as u32)];
            let r = (result[[y, x, 0]] * 255.0).round().clamp(0.0, 255.0) as u8;
            let g = (result[[y, x, 1]] * 255.0).round().clamp(0.0, 255.0) as u8;
            let b = (result[[y, x, 2]] * 255.0).round().clamp(0.0, 255.0) as u8;
            img_buf.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, orig_px[3]]));
        }
    }
    img_buf
        .save(output_path)
        .with_context(|| format!("failed to save output: {}", output_path.display()))?;
    Ok(())
}

fn marker(phase: &str) {
    eprintln!("[LAMA_MARKER] phase {}", phase);
    std::io::stderr().flush().ok();
}
