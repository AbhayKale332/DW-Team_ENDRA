"""Frozen desktop entry point. Reuses V5 tiling, preprocessing and output contracts."""
import argparse
import hmac
import json
import os
from pathlib import Path
import socket
import sys

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Model_Traning" / "v5"))


def validate_model(graph):
    import onnx

    graph = Path(graph).resolve()
    metadata = json.loads(Path(str(graph) + ".json").read_text())
    spec = metadata.get("preproc", {})
    if not all(k in spec for k in ("mean", "std", "tile_size", "canonical_gsd_m")):
        raise ValueError("Missing ONNX preprocessing metadata")
    model = onnx.load(str(graph), load_external_data=False)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                target = (graph.parent / entry.value).resolve()
                if target.parent != graph.parent or not target.is_file():
                    raise ValueError(f"Missing or unsafe ONNX weight sidecar: {entry.value}")
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()
    token = os.environ.get("DW_DESKTOP_TOKEN", "")
    if not token:
        raise ValueError("The desktop launcher must supply DW_DESKTOP_TOKEN")
    print("[desktop] Validating ONNX model", flush=True)
    graph = validate_model(args.onnx)

    print("[desktop] Loading inference dependencies", flush=True)
    from serve.app import create_app, load_backend
    import uvicorn
    from fastapi.responses import JSONResponse

    threads = min(4, os.cpu_count() or 1)
    import torch
    torch.set_num_threads(threads)
    os.environ.setdefault("DW_BATCH_TILES", "1")
    print("[desktop] Loading ONNX session", flush=True)
    load_backend(onnx=str(graph), onnx_providers=["CPUExecutionProvider"], onnx_threads=threads)
    api = create_app()

    @api.middleware("http")
    async def authenticate(request, call_next):
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {token}"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    # Bind once; no gap between choosing a free port and starting the service.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    print(f"DW_READY {sock.getsockname()[1]}", flush=True)
    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", loop="asyncio", http="h11", log_level="warning"))
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()


if __name__ == "__main__":
    main()
