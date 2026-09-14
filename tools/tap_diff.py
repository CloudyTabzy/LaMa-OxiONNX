#!/usr/bin/env python3
"""Tap-level differential harness: OxiONNX vs ONNX Runtime.

Two modes:

  sample   Build a tap-augmented model (every Nth node output promoted to a
           graph output), run it under ONNX Runtime with graph optimisations
           disabled, and dump every output as raw f32 + shape files.

  compare  Load the ORT dump and an OxiONNX dump (produced by the worker with
           ``LAMA_OXIONNX_DUMP_TAPS``), diff tensor by tensor in graph order,
           and report the first divergence.

Full recipe in tools/README.md. Quick version:

  python tools/tap_diff.py sample  --model lama_fp32.onnx \\
      --image test_data/test_image.png --mask test_data/test_mask.png \\
      --out C:/tmp/taps --start 0 --step 1000

  set LAMA_OXIONNX_DUMP_TAPS=C:/tmp/taps/oxi
  set OXIONNX_OPT_LEVEL=none
  set OXIONNX_NO_SESSION_CACHE=1
  target/release/lama-worker-oxionnx.exe --image test_data/test_image.png \\
      --mask test_data/test_mask.png --model C:/tmp/taps/model_taps.onnx \\
      --output C:/tmp/taps/out.png

  python tools/tap_diff.py compare --ort C:/tmp/taps/ort --oxi C:/tmp/taps/oxi
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

TAP_MODEL_NAME = "model_taps.onnx"
SHAPES_NAME = "shapes.json"


def safe_name(name: str) -> str:
    """File-name-safe tensor name; keeps dots so both sides agree."""
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in name)


def pad16_dims(h: int, w: int) -> tuple[int, int, int, int]:
    """Same mod-16 padding as the Rust worker (`pad16_dims`)."""
    ph = -(-h // 16) * 16
    pw = -(-w // 16) * 16
    pt = (ph - h) // 2
    pl = (pw - w) // 2
    return ph, pw, pt, pl


def build_inputs(image_path: str, mask_path: str):
    """Replicate the worker's model inputs exactly: RGB/255 CHW + bool mask CHW,
    both reflect-padded to mod-16."""
    from PIL import Image

    rgba = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
    image = rgba[:, :, :3].astype(np.float32) / 255.0  # HWC
    gray = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    mask = gray > 0  # HW, anything nonzero = inpaint (worker `MASK_THRESHOLD = 0`)

    h, w = image.shape[:2]
    assert mask.shape == (h, w), "image/mask dimensions differ"
    ph, pw, pt, pl = pad16_dims(h, w)
    pb, pr = ph - h - pt, pw - w - pl
    image = np.pad(image, ((pt, pb), (pl, pr), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((pt, pb), (pl, pr)), mode="reflect")

    image_chw = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    mask_chw = np.ascontiguousarray(mask.astype(np.float32)[None, None])
    return image_chw, mask_chw


def cmd_sample(args: argparse.Namespace) -> int:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        print(f"error: sample mode needs `onnx` and `onnxruntime` ({exc})", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    model = onnx.load(args.model)
    nodes = list(model.graph.node)
    existing = {o.name for o in model.graph.output}

    taps = []
    for idx in range(args.start, len(nodes), args.step):
        node = nodes[idx]
        outputs = [o for o in node.output if o]
        if not outputs or outputs[0] in existing:
            continue
        info = model.graph.output.add()
        info.name = outputs[0]
        taps.append(
            {"idx": idx, "op": node.op_type, "name": outputs[0],
             "input": node.input[0] if node.input else ""}
        )

    augmented = os.path.join(args.out, TAP_MODEL_NAME)
    onnx.save(model, augmented)
    with open(os.path.join(args.out, "taps.json"), "w", encoding="utf-8") as fh:
        json.dump(taps, fh, indent=1)
    print(f"graph nodes: {len(nodes)}  taps: {len(taps)}  -> {augmented}")

    image, mask = build_inputs(args.image, args.mask)
    print(f"inputs: image {image.shape}  mask {mask.shape}  masked px: {int(mask.sum())}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(augmented, options, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(names, {"image": image, "mask": mask})

    out_dir = os.path.join(args.out, "ort")
    os.makedirs(out_dir, exist_ok=True)
    shapes = {}
    for name, arr in zip(names, outputs):
        arr = np.asarray(arr, dtype=np.float32)
        with open(os.path.join(out_dir, safe_name(name) + ".bin"), "wb") as fh:
            fh.write(arr.tobytes())
        shapes[name] = list(arr.shape)
    with open(os.path.join(out_dir, SHAPES_NAME), "w", encoding="utf-8") as fh:
        json.dump(shapes, fh, indent=1)
    print(f"ORT outputs: {len(names)}  -> {out_dir}")
    print("next: run the worker with LAMA_OXIONNX_DUMP_TAPS=<dir>/oxi "
          "OXIONNX_OPT_LEVEL=none OXIONNX_NO_SESSION_CACHE=1 on the augmented model")
    return 0


def load_oxi_tensor(directory: str, name: str) -> np.ndarray | None:
    path = os.path.join(directory, safe_name(name))
    if not os.path.exists(path + ".bin") or not os.path.exists(path + ".shape"):
        return None
    with open(path + ".shape", encoding="utf-8") as fh:
        shape = tuple(int(x) for x in fh.read().split())
    return np.fromfile(path + ".bin", dtype=np.float32).reshape(shape)


def cmd_compare(args: argparse.Namespace) -> int:
    taps_path = os.path.join(os.path.dirname(os.path.abspath(args.ort)), "taps.json")
    if not os.path.exists(taps_path):
        taps_path = os.path.join(args.ort, "taps.json")
    with open(taps_path, encoding="utf-8") as fh:
        taps = json.load(fh)
    with open(os.path.join(args.ort, SHAPES_NAME), encoding="utf-8") as fh:
        ort_shapes = json.load(fh)

    rows = []
    first_bad = None
    for tap in taps:
        name = tap["name"]
        ort = np.fromfile(os.path.join(args.ort, safe_name(name) + ".bin"), dtype=np.float32)
        ort = ort.reshape(ort_shapes[name])
        oxi = load_oxi_tensor(args.oxi, name)
        if oxi is None:
            rows.append((tap["idx"], tap["op"], name, None, "MISSING"))
            first_bad = first_bad or (tap["idx"], tap["op"], name)
            continue
        if oxi.shape != ort.shape:
            rows.append((tap["idx"], tap["op"], name, None,
                         f"SHAPE {ort.shape} vs {oxi.shape}"))
            first_bad = first_bad or (tap["idx"], tap["op"], name)
            continue
        diff = np.abs(ort.astype(np.float64) - oxi.astype(np.float64))
        rows.append((tap["idx"], tap["op"], name, float(diff.max()),
                     f"mean {diff.mean():.3g}"))
        if diff.max() > args.tol and first_bad is None:
            first_bad = (tap["idx"], tap["op"], name)

    # Final graph output too, when both dumps carry it.
    output = load_oxi_tensor(args.oxi, "output")
    ort_out = os.path.join(args.ort, safe_name("output") + ".bin")
    if output is not None and os.path.exists(ort_out):
        want = np.fromfile(ort_out, dtype=np.float32).reshape(ort_shapes["output"])
        if want.shape == output.shape:
            diff = np.abs(want.astype(np.float64) - output.astype(np.float64))
            rows.append((-1, "Output", "output", float(diff.max()),
                         f"mean {diff.mean():.3g}"))
            if diff.max() > args.tol and first_bad is None:
                first_bad = (-1, "Output", "output")

    print(f"{'node':>6}  {'op':<14} {'max abs diff':>13}  detail")
    for idx, op, name, diff, detail in rows:
        is_bad = diff is None or diff > args.tol
        flag = "  <-- DIVERGES" if is_bad else ""
        shown = "MISSING" if diff is None else f"{diff:.6g}"
        print(f"{idx:6d}  {op:<14} {shown:>13}  {detail}{flag}")
        if is_bad:
            break

    print()
    if first_bad:
        print(f"FIRST DIVERGENCE: node {first_bad[0]} ({first_bad[1]}) {first_bad[2]}")
        return 1
    print(f"no divergence above tolerance {args.tol:g}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    sample = sub.add_parser("sample", help="augment the model and dump ORT taps")
    sample.add_argument("--model", required=True)
    sample.add_argument("--image", required=True)
    sample.add_argument("--mask", required=True)
    sample.add_argument("--out", required=True)
    sample.add_argument("--start", type=int, default=0)
    sample.add_argument("--step", type=int, default=1000)
    sample.set_defaults(func=cmd_sample)

    compare = sub.add_parser("compare", help="diff ORT and OxiONNX tap dumps")
    compare.add_argument("--ort", required=True)
    compare.add_argument("--oxi", required=True)
    compare.add_argument("--tol", type=float, default=1e-2,
                         help="max abs diff considered noise (default 1e-2; the "
                              "engine's real noise floor is ~1e-6 per tap)")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
