"""Export the trained network to ONNX, for CPU-only standalone deployment.

    python -m infer.export_onnx --ckpt outputs/v4/best.pt --out outputs/v4/depthwizard.onnx

Why this is a deliverable and not a nice-to-have: the rubric awards half the score
to "software stability and successful standalone deployment", and the panel may
well run the demo on a laptop with no GPU and no CUDA-matched PyTorch build.  An
ONNX graph plus onnxruntime is ~50 MB of dependency instead of ~2.5 GB, runs on
CPU, and removes `transformers` (and therefore the hub download and an
HF token) from the deployed machine entirely.

The exported graph takes a fixed `tile_size` window — batch is dynamic, spatial
is not, because the encoder's rotary position embeddings are built for the tile
grid it was traced at.  That is exactly how the model is used anyway:
`infer/engine.py` always feeds it `tile_size` windows and blends the results.

The preprocessing contract travels alongside as `<name>.json`, so an ONNX
deployment reproduces training inputs from the same recipe the checkpoint carries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _HeightOnly(torch.nn.Module):
    """Trim the dict output to the two tensors a deployment needs.

    ONNX has no dictionaries; exporting the full head output would also drag the
    (B, K, H/2, W/2) bin logits into the graph for no reason — they are a training
    signal, not a product.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        out = self.net(image)
        return out["fused"], out["seg"].argmax(1, keepdim=True).to(torch.int32)


def export(ckpt: str, out_path: str, *, opset: int = 18, hf_token: str = "",
           check: bool = True) -> Path:
    from infer.load import load_model

    device = torch.device("cpu")
    model, spec, _cfg = load_model(ckpt, device, hf_token)
    wrapper = _HeightOnly(model).eval()

    s = spec.tile_size
    # Traced at batch 2, NOT batch 1.  `torch.export` treats 0 and 1 as special
    # sizes and will specialise a dimension whose example value is 1 rather than
    # keep it symbolic — so tracing a "dynamic batch" export on a batch-1 example
    # is self-defeating.  The v4 Kaggle run lost its ONNX deliverable to exactly
    # this: both the `dynamic_shapes` attempt and the `dynamic_axes` fallback
    # (which the dynamo exporter converts and re-traces) died with
    #
    #   ConstraintViolationError: Constraints violated (batch)!
    #   - Not all values of batch = L['image'].size()[0] in the specified range
    #     satisfy the generated guard L['image'].size()[0] <= 2
    #   - solving the guards ... resulted in a specialized value of 1
    #
    # and the run finished with no depthwizard.onnx at all.  Two 518x518 fp32
    # tiles is 6 MB of CPU memory to trace with; the graph is identical.
    dummy = torch.zeros(2, 3, s, s, dtype=torch.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(input_names=["image"], output_names=["height_m", "seg"],
                  opset_version=opset)
    with torch.no_grad():
        try:
            # torch >= 2.5 dynamo exporter: `dynamic_shapes` is the supported way
            # to say "batch is free"; `dynamic_axes` still works but warns and is
            # converted internally.  `min=1` is explicit because `Dim`'s own
            # default lower bound is 2, which would make a batch-1 inference —
            # what `infer/engine.py` actually does on a single tile — out of range.
            batch = torch.export.Dim("batch", min=1)
            torch.onnx.export(wrapper, (dummy,), str(out_path),
                              dynamic_shapes={"image": {0: batch}}, **common)
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            print(f"[onnx] dynamic_shapes path unavailable "
                  f"({type(e).__name__}: {str(e).splitlines()[0][:160]}); "
                  "falling back to dynamic_axes")
            torch.onnx.export(
                wrapper, (dummy,), str(out_path), do_constant_folding=True,
                dynamic_axes={"image": {0: "batch"}, "height_m": {0: "batch"},
                              "seg": {0: "batch"}}, **common)
    # `opset` is a *request*.  The dynamo exporter emits at its own native
    # opset and then asks onnxscript to down-convert; when an op has no version
    # adapter that conversion fails, onnxscript prints a traceback, says "the
    # model was not modified", and the export succeeds anyway at the native
    # opset.  That is exactly what the v4-2 run hit —
    #     RuntimeError: adapter_lookup: ... No Adapter To Version $17 for Resize
    # (Resize comes from the head's F.interpolate) — and because
    # `torch.onnx.export` never raised, the run recorded "opset": 17 for a graph
    # that is not opset 17.  Read it back off the model instead of asserting it.
    real_opset, ext = _describe(out_path)
    meta = {"preproc": spec.to_dict(), "input": f"float32 (batch,3,{s},{s}), "
            "encoder-normalised RGB at canonical GSD",
            "outputs": {"height_m": f"float32 (batch,1,{s},{s}) nDSM in metres",
                        "seg": f"int32 (batch,1,{s},{s}) class id"},
            "opset": real_opset, "opset_requested": opset,
            "external_data": [f.name for f in ext],
            "source_checkpoint": str(ckpt)}
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=2))
    mb = out_path.stat().st_size / 1024 ** 2
    print(f"[onnx] {out_path} ({mb:.0f} MB) + {out_path.name}.json"
          + ("" if real_opset == opset else
             f"  [!] opset {real_opset}, not the requested {opset}"))
    if ext:
        # A 300 M-parameter model that serialises to a few MB has put its
        # weights in external .data files next to the graph: the v4-2 run wrote
        # a 4 MB `depthwizard.onnx`.  Shipping that file alone loads nothing,
        # and with --make_zip false nothing else bundles the directory, so the
        # sidecars have to be named out loud.
        tot = sum(f.stat().st_size for f in ext) / 1024 ** 2
        print(f"[onnx] weights are EXTERNAL — ship these too ({tot:.0f} MB): "
              + ", ".join(f.name for f in ext))
    elif mb < 50:
        print(f"[onnx] !! {mb:.0f} MB with no external-data files found — "
              f"the graph looks weightless; check before shipping it.")

    if check:
        # Verified at batch 1, having been traced at batch 2, so the check also
        # proves the batch axis really came out dynamic instead of baked — which
        # is the one property `dynamic_shapes` is there to deliver and the one an
        # `onnxruntime` deployment discovers the hard way if it did not.
        verify(out_path, wrapper, dummy[:1])
    return out_path


def _describe(out_path) -> tuple[int, list]:
    """(actual default opset of the written graph, external weight files).

    The opset is read back off the proto rather than taken from the export
    request, because a failed down-conversion leaves the graph at the exporter's
    native opset without raising.  External-data files are found by asking the
    graph which ones it references, falling back to "siblings that appeared next
    to it" when the initialisers cannot be walked.
    """
    out_path = Path(out_path)
    opset, refs = -1, []
    try:
        import onnx

        m = onnx.load(str(out_path), load_external_data=False)
        opset = next((i.version for i in m.opset_import if i.domain in ("", "ai.onnx")),
                     -1)
        for init in m.graph.initializer:
            for kv in init.external_data:
                if kv.key == "location":
                    refs.append(kv.value)
    except Exception as e:  # noqa: BLE001
        print(f"[onnx] could not introspect the written graph ({e})")
    d = out_path.parent
    if refs:
        ext = [d / r for r in dict.fromkeys(refs) if (d / r).is_file()]
    else:
        ext = sorted(f for f in d.glob(f"{out_path.name}*")
                     if f != out_path and f.suffix != ".json")
    return opset, ext


def verify(onnx_path, torch_module=None, dummy=None, tol: float = 5e-2) -> bool:
    """Run the graph once and, when given the torch module, compare against it.

    A silent export that produces different numbers is worse than no export, so
    this runs by default rather than being an opt-in flag.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[onnx] onnxruntime not installed — graph written but not verified. "
              "`pip install onnxruntime` to check it.")
        return False
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if dummy is None:
        shape = sess.get_inputs()[0].shape
        s = int(shape[-1]) if isinstance(shape[-1], int) else 512
        dummy = torch.zeros(1, 3, s, s)
    got = sess.run(None, {"image": dummy.numpy()})
    print(f"[onnx] ran ok -> height {got[0].shape}, seg {got[1].shape}")
    if torch_module is None:
        return True
    with torch.no_grad():
        ref = torch_module(dummy)[0].numpy()
    import numpy as np

    err = float(np.abs(ref - got[0]).max())
    ok = err <= tol
    print(f"[onnx] max |torch - onnx| = {err:.4f} m  ->  {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="export DepthWizard to ONNX")
    ap.add_argument("--ckpt", default="outputs/v4/best.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    out = a.out or str(Path(a.ckpt).parent / "depthwizard.onnx")
    export(a.ckpt, out, opset=a.opset, hf_token=a.hf_token, check=not a.no_check)


if __name__ == "__main__":
    main()
