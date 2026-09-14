#!lama-oxionnx-gimp-python
"""GIMP 3.x plug-in for LaMa inpainting through the pure-Rust OxiONNX worker.

Exposes a single image-scoped procedure:

- ``plug-in-lama-oxionnx`` — single-pass LaMa FFC model. Runs entirely in
  the bundled Rust sidecar ``lama-worker-oxionnx.exe``, which links the
  vendored OxiONNX engine: no ONNX Runtime, no numpy, no Pillow, no C/C++
  runtime to install.

The GIMP side runs in GIMP's bundled MINGW Python 3.14 (via the per-user
``.interp`` mapping installed by ``install.bat``) and imports only
``gi``/GEGL. Inference is always out of process: this file is the bridge
that exports the drawable + selection to temp PNGs, spawns the worker,
and loads the result into the drawable's shadow buffer.

This plug-in is installed side by side with the ONNX-Runtime-based one
(``plug-in-lama-inpaint``): it uses its own procedure name, menu entry and
plug-in directory, so both can be compared in the same GIMP session.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import gi
gi.require_version('Gegl', '0.4')
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')

from gi.repository import Gegl, Gimp, GimpUi, GLib


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "lama.log"))
# Max number of lines to keep in the log file. Older lines are
# truncated on each write so the log doesn't grow without bound
# across many inpaint runs.
LOG_MAX_LINES = 200

# The pure-Rust OxiONNX sidecar (built by install.bat from this repo).
WORKER_BINARY = os.path.abspath(
    os.path.join(PLUGIN_DIR, "lama-worker-oxionnx.exe")
)
# The dynamic-H/W LaMa export (accepts any mod-16 spatial size), the
# same model the worker was validated against.
MODEL_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "lama_fp32.onnx"))

WORKER_TIMEOUT_SECONDS = 300
WORKER_POLL_INTERVAL_SECONDS = 0.25

# The worker currently always runs the whole image at native resolution
# (no ROI path yet — see the repository roadmap). LaMa's FFC branch is
# quadratic in the token count, so refuse very large images up front
# instead of risking a multi-minute run and multi-GB allocations.
# Overridable with LAMA_OXIONNX_MAX_PIXELS for experiments.
DEFAULT_MAX_PIXELS = 4_000_000


def _log(msg):
    """Append a line to the plug-in log file.

    GIMP does not relay plug-in stdout/stderr to any visible console in
    GUI mode (``G_SPAWN_LEAVE_DESCRIPTORS_OPEN`` inherits GIMP's own
    descriptors, which are null for a normal desktop launch). Writing to
    a file instead guarantees the information is always available.

    To prevent stale results from accumulating across many inpaint
    runs, we truncate the log on each call to keep only the most
    recent ``LOG_MAX_LINES`` entries. This makes the log useful for
    debugging the last few runs without growing without bound.
    """
    try:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        # Read existing content (if any), append the new line, and
        # truncate to the most recent LOG_MAX_LINES entries.
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                existing = f.read().splitlines()
        except OSError:
            existing = []
        existing.append(line.rstrip("\n"))
        if len(existing) > LOG_MAX_LINES:
            existing = existing[-LOG_MAX_LINES:]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(existing) + "\n")
    except OSError:
        pass


def _env_truthy(name):
    """Return True iff the named env var is set to a truthy string."""
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


# Per-run logging is opt-in: LAMA_OXIONNX_LOG=1. Errors are always logged;
# routine lines (worker path, per-run phase profile) only when asked for, so
# a successful inference leaves no trace on disk.
_VERBOSE_LOG = _env_truthy("LAMA_OXIONNX_LOG")


def _vlog(msg):
    """Log a routine per-run line only when ``LAMA_OXIONNX_LOG`` is set."""
    if _VERBOSE_LOG:
        _log(msg)


def _normalize_path(path):
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.strip().strip('"')
    path = os.path.expanduser(os.path.expandvars(path))
    return os.path.abspath(path)


def _max_pixels():
    """Read the optional ``LAMA_OXIONNX_MAX_PIXELS`` guard override."""
    raw = os.environ.get("LAMA_OXIONNX_MAX_PIXELS")
    if raw is None:
        return DEFAULT_MAX_PIXELS
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_PIXELS
    return value if value > 0 else DEFAULT_MAX_PIXELS


def find_worker_binary():
    """Locate the OxiONNX worker next to the plug-in.

    ``LAMA_OXIONNX_WORKER`` may point at a development build; otherwise
    the installed binary next to this file is used. Returns ``None``
    when neither exists so the caller can report a clear install error.
    """
    override = _normalize_path(os.environ.get("LAMA_OXIONNX_WORKER"))
    if override and os.path.isfile(override):
        return override
    if os.path.isfile(WORKER_BINARY):
        return WORKER_BINARY
    return None


def _process_detail(completed):
    """Summarise a worker failure for the error dialog (last stderr line)."""
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return f"exit code {completed.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = lines[-1] if lines else output
    return detail[:500]


def _windows_no_window_popen_kwargs():
    """Build kwargs that hide the spawned worker console on Windows.

    Both ``CREATE_NO_WINDOW`` and a ``STARTUPINFO`` with
    ``STARTF_USESHOWWINDOW``/``SW_HIDE`` are applied because some Python
    builds on Windows expose one flag but not the other. This only hides
    the *spawned ML worker* console; the terminal used to launch GIMP
    with ``--verbose`` remains visible.
    """
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 0,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name != "nt":
        return kwargs

    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    startupinfo = None
    start_flags = getattr(subprocess, "STARTF_USESHOWWINDOW", None)
    sw_hide = getattr(subprocess, "SW_HIDE", None)
    if start_flags is not None and sw_hide is not None:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = start_flags
        startupinfo.wShowWindow = sw_hide
    if startupinfo is not None:
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _run_worker_with_progress(command, progress_callback):
    """Run the ML worker with ``Popen`` so the GIMP UI can stay responsive.

    Pipe draining is delegated to a daemon reader thread that reads the
    merged stdout/stderr line-by-line into a ``queue.Queue``. The main
    thread never blocks on the pipe directly: the worker emits only a
    few short marker lines and then spends its time in inference, so a
    direct read could sit waiting for the next chunk that does not
    arrive until the worker exits — that would freeze progress updates,
    the timeout check, and the ``finally`` cleanup all at once. With the
    reader thread the main thread can poll ``process.poll()``
    continuously, drive the GIMP progress callback every ~0.25 s, and
    still enforce the 300-second deadline.

    * stdout and stderr are merged into a single pipe (``stderr=STDOUT``).
    * The reader thread is a daemon so it cannot block process exit; it
      closes the pipe handle itself in ``finally`` and signals completion
      with a ``None`` sentinel.
    * On timeout/exit the child is terminated and then killed if it
      does not exit, so the pipe closes and the reader thread can
      drain. The reader is then joined with a brief timeout.
    * Returns a ``SimpleNamespace`` with ``returncode``, ``stdout``,
      ``stderr`` and ``timed_out`` attributes — a
      ``subprocess.CompletedProcess``-like shape that the error path
      consumes.
    * Preserves UTF-8 decoding with ``errors="replace"`` and the
      Windows hide-console flags from
      :func:`_windows_no_window_popen_kwargs`.
    """
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    popen_kwargs = _windows_no_window_popen_kwargs()
    process = subprocess.Popen(command, **popen_kwargs)

    output_queue: "queue.Queue[object]" = queue.Queue()

    def _reader():
        """Drain the merged pipe line-by-line into ``output_queue``.

        ``readline`` blocks between lines, which is fine because this
        thread is the only consumer of the pipe and is daemonic. When
        the worker exits (or is terminated/killed) the pipe closes,
        ``readline`` returns ``""``, the iterator terminates, and the
        ``finally`` block closes the pipe handle and signals completion
        with a ``None`` sentinel.
        """
        if process.stdout is None:
            output_queue.put(None)
            return
        try:
            for line in iter(process.stdout.readline, ""):
                output_queue.put(line)
        except Exception:
            pass
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass
            output_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = False
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                break

            if progress_callback is not None:
                try:
                    progress_callback()
                except Exception:
                    pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            time.sleep(min(WORKER_POLL_INTERVAL_SECONDS, remaining))
    finally:
        # Stop the child first so the pipe closes and the reader thread
        # can drain and exit on its own, then join the reader briefly.
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass

        reader_thread.join(timeout=2.0)

    output_chunks = []
    while True:
        try:
            item = output_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            break
        output_chunks.append(item)

    return SimpleNamespace(
        returncode=process.returncode,
        stdout="".join(output_chunks),
        stderr=None,
        timed_out=timed_out,
    )


def save_buffer_as_png(buffer, path):
    """Save a GEGL buffer through buffer-source -> png-save."""
    graph = Gegl.Node()
    source = graph.create_child("gegl:buffer-source")
    source.set_property("buffer", buffer)
    saver = graph.create_child("gegl:png-save")
    saver.set_property("path", path)
    source.link(saver)
    saver.process()
    if not os.path.isfile(path):
        raise OSError(f"GEGL did not create {path}")


def save_drawable_selection_mask(image, drawable, width, height, path):
    """Copy the image-space selection into a drawable-local Y u8 buffer."""
    selection = image.get_selection()
    if selection is None:
        raise RuntimeError("GIMP did not return the active selection")

    success, offset_x, offset_y = drawable.get_offsets()
    if not success:
        raise RuntimeError("Could not read the drawable offset")

    selection_buffer = selection.get_buffer()
    local_mask = Gegl.Buffer.new("Y u8", 0, 0, width, height)
    source_rect = Gegl.Rectangle.new(offset_x, offset_y, width, height)
    destination_rect = Gegl.Rectangle.new(0, 0, width, height)
    selection_buffer.copy(
        source_rect,
        Gegl.AbyssPolicy.NONE,
        local_mask,
        destination_rect,
    )
    save_buffer_as_png(local_mask, path)


def load_png_into_shadow(drawable, path, expected_width, expected_height):
    """Load a PNG through png-load -> write-buffer into the shadow buffer."""
    shadow_buffer = drawable.get_shadow_buffer()
    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", path)

    bounds = loader.get_bounding_box()
    if bounds.width != expected_width or bounds.height != expected_height:
        raise ValueError(
            "worker result dimensions differ: "
            f"{bounds.width}x{bounds.height} vs "
            f"{expected_width}x{expected_height}"
        )

    writer = graph.create_child("gegl:write-buffer")
    writer.set_property("buffer", shadow_buffer)
    loader.link(writer)
    writer.process()
    shadow_buffer.flush()


def _return_error(procedure, status, message):
    error = GLib.Error.new_literal(Gimp.PlugIn.error_quark(), message, 0)
    return procedure.new_return_values(status, error)


def _calling_error(procedure, message):
    return _return_error(procedure, Gimp.PDBStatusType.CALLING_ERROR, message)


def _execution_error(procedure, message):
    return _return_error(procedure, Gimp.PDBStatusType.EXECUTION_ERROR, message)


def _safe_progress(callable_, *args, **kwargs):
    """Invoke a Gimp.progress_* function without aborting on failure.

    GIMP's progress API is straightforward in normal use, but wrapping
    every call keeps the plug-in functional if GIMP throws or if the
    PDB is in a non-interactive transition.
    """
    try:
        return callable_(*args, **kwargs)
    except Exception:
        return None


class LamaOxiONNX(Gimp.PlugIn):
    def do_set_i18n(self, _name):
        return False

    def do_query_procedures(self):
        return ["plug-in-lama-oxionnx"]

    def do_create_procedure(self, name):
        if name == "plug-in-lama-oxionnx":
            Gegl.init(None)
            procedure = Gimp.ImageProcedure.new(
                self,
                name,
                Gimp.PDBProcType.PLUGIN,
                self.run,
                None,
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("_LaMa Inpaint (OxiONNX)...")
            procedure.set_icon_name(GimpUi.ICON_GEGL)
            procedure.add_menu_path("<Image>/Filters/Enhance/")

            procedure.set_documentation(
                "Inpaint the active selection with the LaMa model "
                "(pure-Rust OxiONNX sidecar, single-pass).",
                "Exports the drawable and selection to the bundled Rust "
                "OxiONNX worker, then applies the inpainted result to the "
                "active selection. Color preservation is exact outside the "
                "selection; best for removing spots, wires, and small "
                "blemishes.",
                name,
            )
            procedure.set_attribution(
                "LaMa OxiONNX Plug-in",
                "LaMa OxiONNX Plug-in",
                "2026",
            )
            return procedure
        return None

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        try:
            return self._run_lama(procedure, run_mode, image, drawables)
        except Exception as exc:
            _log(f"run EXCEPTION: {exc}")
            return _execution_error(procedure, f"LaMa Inpaint (OxiONNX) error: {exc}")

    # ----------------- LaMa backend -----------------

    def _run_lama(self, procedure, run_mode, image, drawables):
        # Progress is started exactly once after the cheap pre-flight
        # checks have all passed, and is always ended in ``finally`` so
        # the GIMP progress bar cannot be left in a half-state on any
        # failure path (timeout, exception, cancelled run).
        #
        # Wall-clock profiling: every phase of the round trip is timed
        # (export, mask, worker, import, apply) and written to the log as
        # `profile: bridge ...` / `profile: worker ...` lines when the run
        # completes. GIMP's own repaint after `displays_flush()` happens in
        # its main loop and is not observable from here.
        t_run_start = time.monotonic()
        progress_started = False

        def _phase(text, fraction):
            if not progress_started:
                return
            _safe_progress(Gimp.progress_set_text, text)
            _safe_progress(Gimp.progress_update, fraction)

        def _pulse():
            if not progress_started:
                return
            _safe_progress(Gimp.progress_pulse)
            _safe_progress(
                Gimp.progress_set_text,
                "Running LaMa OxiONNX inference (pure Rust)... please wait",
            )

        def _worker_progress_callback():
            _pulse()

        try:
            if len(drawables) != 1:
                return _calling_error(
                    procedure,
                    f"LaMa Inpaint (OxiONNX) requires exactly one drawable; "
                    f"got {len(drawables)}.",
                )

            drawable = drawables[0]
            intersects, selection_x, selection_y, selection_width, selection_height = (
                drawable.mask_intersect()
            )
            if not intersects:
                return _calling_error(
                    procedure,
                    "Make a non-empty selection on the active drawable first.",
                )

            width = drawable.get_width()
            height = drawable.get_height()
            if width <= 0 or height <= 0:
                return _calling_error(procedure, "The active drawable is empty.")

            max_pixels = _max_pixels()
            if width * height > max_pixels:
                return _calling_error(
                    procedure,
                    f"The OxiONNX worker currently supports images up to "
                    f"{max_pixels:,} pixels at native resolution "
                    f"(this image is {width * height:,}).\n"
                    "Use the ONNX Runtime plug-in for larger images, or set "
                    "LAMA_OXIONNX_MAX_PIXELS to override the guard.",
                )

            if not os.path.isfile(MODEL_PATH):
                return _calling_error(
                    procedure,
                    f"Model is missing: {MODEL_PATH}. Reinstall the plug-in.",
                )

            worker_binary = find_worker_binary()
            if worker_binary is None:
                return _calling_error(
                    procedure,
                    f"The OxiONNX worker is missing: {WORKER_BINARY}.\n"
                    "Run install.bat from gimp\\ in the repository, or build "
                    "with:\n"
                    "  cargo build --release\n"
                    "then copy target\\release\\lama-worker-oxionnx.exe next "
                    "to this plug-in.",
                )

            # All pre-flight checks passed; start the progress bar.
            _safe_progress(Gimp.progress_init, "LaMa Inpaint (OxiONNX)")
            progress_started = True
            _phase("Preparing image...", 0.05)

            try:
                with tempfile.TemporaryDirectory(prefix="gimp-lama-") as temp_dir:
                    image_path = os.path.join(temp_dir, "image.png")
                    mask_path = os.path.join(temp_dir, "mask.png")
                    output_path = os.path.join(temp_dir, "result.png")

                    _phase("Preparing image...", 0.10)
                    t = time.monotonic()
                    save_buffer_as_png(drawable.get_buffer(), image_path)
                    export_ms = int((time.monotonic() - t) * 1000)

                    _phase("Exporting mask...", 0.20)
                    t = time.monotonic()
                    save_drawable_selection_mask(
                        image,
                        drawable,
                        width,
                        height,
                        mask_path,
                    )
                    mask_ms = int((time.monotonic() - t) * 1000)

                    command = [
                        worker_binary,
                        "--image",
                        image_path,
                        "--mask",
                        mask_path,
                        "--output",
                        output_path,
                        "--model",
                        MODEL_PATH,
                    ]

                    _phase("Starting LaMa worker...", 0.30)
                    _vlog(f"worker: {worker_binary}")

                    try:
                        t = time.monotonic()
                        completed = _run_worker_with_progress(
                            command, _worker_progress_callback
                        )
                        worker_wall_ms = int((time.monotonic() - t) * 1000)
                    except OSError as exc:
                        return _execution_error(
                            procedure,
                            f"Could not start the LaMa worker: {exc}",
                        )

                    if completed.timed_out:
                        return _execution_error(
                            procedure,
                            "The LaMa worker timed out after "
                            f"{WORKER_TIMEOUT_SECONDS} seconds.",
                        )
                    if completed.returncode != 0:
                        return _execution_error(
                            procedure,
                            "The LaMa worker failed: "
                            + _process_detail(completed),
                        )
                    if not os.path.isfile(output_path):
                        return _execution_error(
                            procedure,
                            "The LaMa worker did not create its output PNG.",
                        )

                    # Log the worker's own phase breakdown (decode,
                    # preprocess, session load, inference, compose, PNG
                    # write). Older workers only emit the shorter `timing`
                    # line; fall back to it.
                    timing_line = None
                    for line in (completed.stdout or "").splitlines():
                        if "[LAMA_MARKER] profile" in line:
                            _vlog(
                                "profile: worker "
                                + line.split("profile", 1)[1].strip()
                            )
                            break
                        if "[LAMA_MARKER] timing" in line:
                            timing_line = line.strip()
                    else:
                        if timing_line:
                            _vlog(timing_line)

                    # Optional debug copies: keeps the exact PNGs exchanged
                    # with the worker so the pipeline can be inspected
                    # without the transient temp directory.
                    debug_dir = os.environ.get("LAMA_OXIONNX_DEBUG_DIR")
                    if debug_dir:
                        try:
                            os.makedirs(debug_dir, exist_ok=True)
                            for src_path, name in (
                                (image_path, "image.png"),
                                (mask_path, "mask.png"),
                                (output_path, "result.png"),
                            ):
                                if os.path.isfile(src_path):
                                    shutil.copyfile(
                                        src_path, os.path.join(debug_dir, name)
                                    )
                            _vlog(f"debug copies written to {debug_dir}")
                        except OSError as exc:
                            _vlog(f"debug copy failed: {exc}")

                    _phase("Applying result...", 0.90)
                    t = time.monotonic()
                    load_png_into_shadow(drawable, output_path, width, height)
                    import_ms = int((time.monotonic() - t) * 1000)

                t = time.monotonic()
                drawable.merge_shadow(True)
                drawable.update(
                    selection_x,
                    selection_y,
                    selection_width,
                    selection_height,
                )
                Gimp.displays_flush()
                apply_ms = int((time.monotonic() - t) * 1000)

                _vlog(
                    "profile: bridge export_ms=%d mask_ms=%d worker_wall_ms=%d "
                    "import_ms=%d apply_ms=%d total_ms=%d"
                    % (
                        export_ms,
                        mask_ms,
                        worker_wall_ms,
                        import_ms,
                        apply_ms,
                        int((time.monotonic() - t_run_start) * 1000),
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                return _execution_error(procedure, f"LaMa image transfer failed: {exc}")
            except Exception as exc:
                return _execution_error(procedure, f"LaMa Inpaint failed: {exc}")

            _phase("Complete", 1.0)
            return procedure.new_return_values(
                Gimp.PDBStatusType.SUCCESS,
                GLib.Error(),
            )
        finally:
            if progress_started:
                _safe_progress(Gimp.progress_end)


Gimp.main(LamaOxiONNX.__gtype__, sys.argv)
