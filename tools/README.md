# Tap-level differential harness — OxiONNX vs ONNX Runtime

`tap_diff.py` is the tool that found the fork's ConvTranspose and small-M
conv bugs (see the [correctness audit](../docs/OXIONNX_REPORT.md#10-correctness-audit-2026-09-14)).
It compares the two engines **tensor by tensor** instead of image by image:
every N-th node output is promoted to a model graph output ("tap"), both
engines run the augmented model on identical inputs with graph
optimisations disabled, and the tensors are diffed in evaluation order. The
first divergent tap is the bug; the diff pattern (which rows, columns,
parities) usually names it.

Use it for **any kernel/op change or engine-version bump** — never validate
a kernel by comparing the fork against its own previous build (that only
proves the change is output-preserving, not correct).

## Requirements

- Python 3.10+ with `onnx`, `onnxruntime`, `numpy`, `Pillow`
- The worker binary built (`cargo build --release` in the repository root)
- The model file (`lama_fp32.onnx`) and an image/mask pair

## Recipe

**1. Build the tap-augmented model and run ONNX Runtime** (optimisations
disabled, exactly as the engine runs it):

```bash
python tools/tap_diff.py sample \
    --model  "C:/.../lama_fp32.onnx" \
    --image  test_data/test_image.png \
    --mask   test_data/test_mask.png \
    --out    C:/tmp/taps \
    --start 0 --step 1000
```

Outputs: `C:/tmp/taps/model_taps.onnx`, `taps.json`, and `ort/` with every
output as raw f32 + shapes. `--start/--step` choose the tap density; use a
dense window (`--start 17400 --step 1`) once a region is implicated.

**2. Run the worker on the same augmented model**, dumping every graph
output:

```bat
set LAMA_OXIONNX_DUMP_TAPS=C:/tmp/taps/oxi
set OXIONNX_OPT_LEVEL=none
set OXIONNX_NO_SESSION_CACHE=1
target\release\lama-worker-oxionnx.exe ^
    --image test_data\test_image.png ^
    --mask   test_data\test_mask.png ^
    --model  C:/tmp/taps/model_taps.onnx ^
    --output C:/tmp/taps/out.png
```

`OXIONNX_OPT_LEVEL=none` keeps node boundaries identical to the ORT run;
the session cache is off by default — if you enabled it with
`OXIONNX_SESSION_CACHE=1`, set `OXIONNX_NO_SESSION_CACHE=1` here so no
cached graph is loaded instead.

**3. Compare:**

```bash
python tools/tap_diff.py compare --ort C:/tmp/taps/ort --oxi C:/tmp/taps/oxi
```

Every tap is printed up to and including the first divergence; the exit code
is 1 when a tap exceeds the tolerance (usable in CI).

## Reading the result

- **Tolerance**: float accumulation order alone gives ~1e-6 per tap across
  this 17k-node graph; the final 8-bit composite can differ by ≤1 LSB.
  Anything ≥1e-2 is a bug. Set `--tol` to tighten or relax (default 1e-2).
- **First divergent node**: the node *producing* that output is the suspect.
  Note that with a tap every N nodes, the actual first bad node can be up to
  N-1 nodes earlier — re-run with a dense window to pin it down.
- **Diff pattern**: errors on all odd rows/columns mean a parity/tap-table
  bug; errors only on the last row/column mean a boundary/out-of-bounds bug;
  errors everywhere at ~10% magnitude mean a weight-layout/index bug.
- **Confirm by hand**: extract the node's input tensors and weights
  (`onnx.numpy_helper.to_array`) and compute the expected values with numpy;
  the audit's scripts are good templates for this step.

## Notes

- The harness replicates the worker's input preparation exactly: RGBA→RGB
  /255, mask `> 0`, reflect padding to mod-16 (the same `pad16_dims`), CHW.
  If you change the worker's preprocessing, update `build_inputs()` to
  match, or the taps beyond the input nodes will legitimately differ.
- Both runs use optimisations off. Comparing optimised-ORT against
  optimised-OxiONNX would still work, but fused nodes change intermediate
  values slightly and add noise to every tap.
- The worker-side dump (`LAMA_OXIONNX_DUMP_TAPS`) writes every graph output
  as `<name>.bin` (little-endian f32) + `<name>.shape`; the harness uses the
  same file-name sanitisation on both sides.
