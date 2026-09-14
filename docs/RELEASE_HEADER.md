# LaMa OxiONNX {{TAG}} — GIMP 3.2 plug-in (Windows x64)

Pure-Rust OxiONNX LaMa inpainting worker. The model runs entirely in Rust —
no ONNX Runtime, no Python packages, roughly 1.8–2x faster than the ONNX
Runtime worker with output matching it to ≤1 LSB.

## Install

1. Download `lama-oxionnx-{{TAG}}-win64.zip` below and unzip it.
2. Run `install.bat` (per-user install; GIMP's own files are untouched).
3. Restart GIMP → **Filters → Enhance → LaMa Inpaint (OxiONNX)...**

The installer downloads the LaMa model (~198 MB, dynamic H/W export) on
first use unless it finds a local copy.

## Requirements

- GIMP 3.2 (64-bit), Windows x64
- No Python, ONNX Runtime or Rust toolchain needed with this bundle
- AVX2/FMA used when available; scalar fallbacks for older x86-64 CPUs

## Verify

`lama-oxionnx-{{TAG}}-win64.zip` SHA-256:
`{{SHA256}}`

## Notes

- Installs side by side with the ONNX Runtime plug-in
  (`plug-ins\lama-inpaint` keeps working); both menu entries can be compared
  in one GIMP session.
- No session cache is used by default; `OXIONNX_SESSION_CACHE=1` opts in to
  a ~373 MB cache that saves only ~0.1 s per run.
- Full details, benchmarks and the correctness audit:
  [README](https://github.com/CloudyTabzy/LaMa-OxiONNX#readme).
