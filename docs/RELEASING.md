# Releasing LaMa OxiONNX

Two things are published from this repository:

1. **The GIMP plug-in bundle** — one zip per release, built by CI
   ([`.github/workflows/release.yml`](../.github/workflows/release.yml)).
2. **The LaMa model** — a permanent, file-once release that the installer
   downloads (`model-lama-fp32-v1`, created once; see below).

## Cutting a plug-in release

```bash
# 1. Make sure master is green and the version is right.
cargo build --release && cargo test --release -p oxionnx-ops --lib

# 2. Tag and push — CI builds, tests, packages and publishes.
git tag v0.1.0
git push origin v0.1.0
```

The workflow runs on `windows-latest`: builds `lama-worker-oxionnx.exe`,
runs the full engine suite (`vendor/oxionnx`, 955 tests), assembles the
bundle from `gimp/` + `LICENSE` + the binary, writes `SHA256SUMS.txt`, and
creates the GitHub release with `--generate-notes`.

Watch it locally:

```bash
gh run watch
gh release view v0.1.0
```

## Bundle contents

`lama-oxionnx-vX.Y.Z-win64.zip` (flat layout — `install.bat` works from it
as-is):

| File | Source |
|---|---|
| `install.bat`, `lama-oxionnx.py`, `gimp-verbose.bat`, `INSTALL.txt` | `gimp/` |
| `lama-worker-oxionnx.exe` | `cargo build --release` |
| `LICENSE` | repository root |

## The model release (created once)

The installer downloads the model from:

```
https://github.com/CloudyTabzy/LaMa-OxiONNX/releases/download/model-lama-fp32-v1/lama_fp32.onnx
```

That release holds the **dynamic-H/W** LaMa export (208 MB, SHA-256
`aacebdf7ced83c4863d7900ed61292224fc718acece717ce2d25630bf44ed584`).
Its notes carry the full provenance: LaMa (Samsung AI Center Toronto /
`advimman/lama`, Apache-2.0) and the ONNX export (Carve, Apache-2.0) —
this file is the Carve export re-serialized with dynamic spatial dims, all
weights byte-identical.

If the model ever changes:

1. Create a **new** permanent release (`model-lama-fp32-v2`) and upload the
   file — do not delete v1 until no installer in the wild references it.
2. Update `MODEL_URL` in [`gimp/install.bat`](../gimp/install.bat).
3. Update the SHA-256 in `gimp/INSTALL.txt` and this document.
4. Re-run the tap-level differential harness against ONNX Runtime before
   tagging a plug-in release (`tools/README.md`); a model swap invalidates
   every previous verification.

## Session cache policy

The worker's on-disk session cache (`OXIONNX_SESSION_CACHE=1`) is **opt-in
and never bundled**: it costs ~373 MB for ~0.1 s saved per run, and it is
keyed to the binary's `SESSION_CACHE_REVISION`. Users who want it build it
on first run. Do not add it to release assets.

## Post-release check (5 minutes)

1. Download the zip from the release page on a machine that has never seen
   the plug-in (or after removing `plug-ins\lama-oxionnx\` and the
   `lama-oxionnx-gimp-python*.interp` files).
2. Unzip, run `install.bat`, restart GIMP, run the filter on a photo.
3. Check `lama.log` shows the per-run `profile:` lines (with
   `LAMA_OXIONNX_LOG=1` set — errors are always logged) and the result is
   inpainted content (not white — see the report's §10 for what that
   looked like when it went wrong).
