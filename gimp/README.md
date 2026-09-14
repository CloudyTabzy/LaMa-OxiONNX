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

The plug-in writes a rotating 200-line log to `lama.log` next to itself
(`%APPDATA%\GIMP\3.2\plug-ins\lama-oxionnx\lama.log`). The log records the
worker path, the worker's `[LAMA_MARKER]` timing line, and any error path.

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
