# GIMP front-end — LaMa OxiONNX plug-in

This directory is the GIMP 3.2 side of the project: a Python bridge that
registers **Filters → Enhance → LaMa Inpaint (OxiONNX)...** and drives the
Rust worker (`lama-worker-oxionnx.exe`) out of process. It installs **side by
side** with the ONNX Runtime plug-in (`plug-in-lama-inpaint`) — own procedure
name, own menu entry, own plug-in directory — so both can be compared in the
same GIMP session.

The worker/OxiONNX engine lives in the repository root; this half is only
GEGL glue: export the drawable + selection mask to temp PNGs, spawn the
worker with a progress-driven UI, load the result into the drawable's shadow
buffer. No numpy, no onnxruntime, no ML wheels — the plug-in runs in GIMP's
bundled MINGW Python (interpreter alias `lama-oxionnx-gimp-python`, written
by the installer).

## Files

| File | Role |
|---|---|
| `lama-oxionnx.py` | the plug-in: procedure registration, GEGL export/import, worker spawn, progress, log rotation |
| `install.bat` | per-user installer: interpreter alias, model + cache copy, `cargo build`, worker copy |
| `gimp-verbose.bat` | launch GIMP with a console and print the plug-in log afterwards |

See the root [`README.md`](../README.md) for install/usage. Quick version:

```bat
cd gimp
install.bat
```

then restart GIMP and use **Filters → Enhance → LaMa Inpaint (OxiONNX)...**.

## Environment overrides

| Variable | Effect |
|---|---|
| `LAMA_OXIONNX_WORKER` | run a development worker build instead of the installed one |
| `LAMA_OXIONNX_DEBUG_DIR` | keep copies of the exchanged `image.png` / `mask.png` / `result.png` |
| `LAMA_OXIONNX_MAX_PIXELS` | override the 4 MP native-resolution guard (default 4,000,000) |
| `OXIONNX_SESSION_CACHE=1` | enable the worker's optional ~373 MB session cache (set before GIMP starts; the worker inherits it — off by default, saves ~0.1 s/run) |

## Logging and per-run profiling

Errors are always logged to a rotating 200-line `lama.log` next to the
plug-in (`%APPDATA%\GIMP\3.2\plug-ins\lama-oxionnx\lama.log`); a successful
run leaves the file untouched. To record the per-run phase profile as well,
set `LAMA_OXIONNX_LOG=1` before starting GIMP. With it enabled, every
successful run adds a `worker:` line plus two `profile:` lines that account
for the whole round trip:

```
[00:02:56] profile: worker decode_ms=18 preprocess_ms=1 load_ms=305 run_ms=8886 compose_ms=5 write_ms=9 total_ms=9227
[00:02:57] profile: bridge export_ms=122 mask_ms=23 worker_wall_ms=9643 import_ms=54 apply_ms=164 total_ms=10034
```

- **worker** (inside the Rust process): PNG decode, preprocessing, session
  load (only when the optional cache is enabled — off by default, so this is
  the model parse + optimize cost), inference (`run_ms`), output compose,
  PNG write.
- **bridge** (inside GIMP): drawable export through GEGL, selection-mask
  export, the worker's full wall time (spawn → exit), result import into the
  shadow buffer, and shadow merge + `displays_flush`.
- `worker_wall_ms − worker.total_ms` is process startup/teardown plus the
  progress-poll interval — the part that is neither engine nor GIMP work.
- `bridge.total_ms` is the plug-in's wall time from entry to just after
  `displays_flush()`. The canvas repaint is *not* included: `gimp-displays-flush`
  only invalidates the render region (`gimp_display_flush` →
  `gimp_display_shell_render_invalidate_area` in the GIMP source), and actual
  rendering happens asynchronously in GIMP's main loop.

Any error path is logged with the reason (worker exit, timeout, dimension
mismatch, missing model/worker) before the filter returns to GIMP.

## Debugging

**Keep the exchanged files.** Set `LAMA_OXIONNX_DEBUG_DIR` for a run and
inspect what the worker actually received and produced. When the plug-in
returned a white rectangle, these three PNGs proved in one run that the
export and mask were correct while the worker's model output was saturated —
see [`docs/OXIONNX_REPORT.md` §10](../docs/OXIONNX_REPORT.md#10-correctness-audit-2026-09-14).

**Run the whole path headlessly.** GIMP's console build can execute the
filter in batch mode, which is how every integration change here was tested
without a GUI:

```bat
"C:\Program Files\GIMP 3\bin\gimp-console-3.2.exe" ^
    --new-instance --console-messages ^
    --batch-interpreter=plug-in-script-fu-eval ^
    --batch "(load \"C:/path/to/script.scm\")" --quit
```

with a script such as:

```scheme
(let* (
  (image (car (file-png-load RUN-NONINTERACTIVE
             "C:/Dev/GIMP_Native_Plugin/LaMa-OxiONNX/test_data/test_photo.png"
             "test_photo.png")))
  (layer (vector-ref (car (gimp-image-get-layers image)) 0))
)
  (gimp-image-select-rectangle image CHANNEL-OP-REPLACE 100 100 150 150)
  (plug-in-lama-oxionnx RUN-NONINTERACTIVE image (vector layer))
  (gimp-image-delete image)
)
```

GIMP 3 API notes learned the hard way while building this test:

- **Drawables are passed as a vector** (`(vector layer)`) — a bare drawable
  fails with *"expected type: vector for argument 3"*.
- **`gimp-image-get-active-drawable` no longer exists**; use
  `(car (gimp-image-get-layers image))` / `(gimp-image-get-layers ...)`.
- **`gimp-layer-new` signature changed** to `(image, name, width, height,
  type)` (name moved to second position).
- `file-png-load` still takes `(run-mode, path, raw-path)`.
- Failure modes are visible as scheme execution errors; success is silent,
  so emit `(gimp-message "...")` markers in the script.

**More GIMP-side lessons** (interpreter `.interp` handoff, subprocess pipe
deadlocks, procedure registration pitfalls) are collected in
[`docs/GIMP_NOTES.md`](../docs/GIMP_NOTES.md), adopted from the ONNX Runtime
plug-in where most of these mistakes were first made.

## Contract with the worker

The bridge and the worker share a strict interface — keep them in sync:

- CLI: `--image --mask --output --model`, all required.
- Input: RGBA PNG (alpha preserved), grayscale mask PNG; **same dimensions**.
- Mask semantics: any nonzero pixel is inpainted; the composite uses the
  soft (0–255) mask for blending.
- Output: RGBA PNG, original alpha bytes, model output **divided by 255**
  before compositing (the reference workers do this; forgetting it produced
  the white rectangle — see report §10).
- Markers: `[LAMA_MARKER] phase ...` and `[LAMA_MARKER] timing ...` on
  stderr; the bridge parses the timing line for the log.
