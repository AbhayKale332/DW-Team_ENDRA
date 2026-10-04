"""Exercise the real desktop API and tiling with a tiny deterministic ONNX graph."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable")
    parser.add_argument("--model", help="Use an actual exported model instead of the deterministic fixture")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dw-inference-test-") as directory:
        root = Path(directory)
        graph = root / "test.onnx"
        nodes = [helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1], keepdims=1),
                 helper.make_node("Mul", ["mean", "zero"], ["zeros"]),
                 helper.make_node("Add", ["zeros", "one"], ["height_m"]),
                 helper.make_node("Cast", ["height_m"], ["seg"], to=TensorProto.INT32),
                 helper.make_node("Mul", ["height_m", "quarter"], ["height_std_m"])]
        constants = [helper.make_tensor(name, TensorProto.FLOAT, [1], [value])
                     for name, value in [("zero", 0.0), ("one", 1.0), ("quarter", 0.25)]]
        model = helper.make_model(helper.make_graph(nodes, "desktop-smoke",
                    [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 16, 16])],
                    [helper.make_tensor_value_info(name, kind, ["batch", 1, 16, 16])
                     for name, kind in [("height_m", TensorProto.FLOAT), ("seg", TensorProto.INT32), ("height_std_m", TensorProto.FLOAT)]],
                    initializer=constants), opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        onnx.save(model, graph)
        Path(str(graph) + ".json").write_text(json.dumps({"preproc": {"mean": [0, 0, 0], "std": [1, 1, 1], "tile_size": 16, "patch": 16, "canonical_gsd_m": 0.5, "radiometric_stretch": False}}))
        if args.model:
            graph = Path(args.model).resolve()
        command = [args.executable] if args.executable else [sys.executable, str(Path(__file__).with_name("inference.py"))]
        environment = {**os.environ, "DW_DESKTOP_TOKEN": "smoke-test-token", "DW_JOBS_DIR": str(root / "jobs"), "DW_CKPT": "", "DW_ONNX": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PROJ_NETWORK": "OFF", "PYTHONUNBUFFERED": "1"}
        with open(root / "stdout.log", "w+") as stdout, open(root / "stderr.log", "w+") as stderr:
            proc = subprocess.Popen(command + ["--onnx", str(graph)], stdout=stdout, stderr=stderr, env=environment)
            try:
                deadline = time.monotonic() + 120
                port = None
                while time.monotonic() < deadline:
                    stdout.seek(0)
                    for line in stdout.read().splitlines():
                        if line.startswith("DW_READY "):
                            port = int(line.split()[1])
                    if port:
                        break
                    if proc.poll() is not None:
                        stderr.seek(0)
                        raise RuntimeError(stderr.read())
                    time.sleep(0.2)
                assert port, "Inference runtime did not start in 120 seconds"
                origin = f"http://127.0.0.1:{port}"

                def fetch(path, body=None, content_type=None, authenticated=True):
                    headers = {"Authorization": "Bearer smoke-test-token"} if authenticated else {}
                    if content_type:
                        headers["Content-Type"] = content_type
                    with urlopen(Request(origin + path, data=body, headers=headers), timeout=30) as response:
                        return response.read()

                for _ in range(100):
                    try:
                        assert json.loads(fetch("/api/health"))["ok"]
                        break
                    except URLError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Inference health endpoint did not become ready")
                try:
                    fetch("/api/health", authenticated=False)
                    raise AssertionError("Unauthenticated backend access must fail")
                except HTTPError as error:
                    assert error.code == 401
                image = io.BytesIO()
                Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(image, format="PNG")
                boundary = "depthwizard-smoke"
                body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="image.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                        + image.getvalue() + f"\r\n--{boundary}--\r\n".encode())
                job = json.loads(fetch("/api/predict", body, f"multipart/form-data; boundary={boundary}"))["job"]
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    status = json.loads(fetch(f"/api/job/{job}"))
                    assert status["stage"] != "error", status.get("error")
                    if status["stage"] == "done":
                        break
                    time.sleep(0.2)
                assert status["stage"] == "done", status
                heights = np.load(io.BytesIO(fetch(f"/api/result/{job}/ndsm_m.npy")))
                assert heights.shape == (32, 32)
                assert np.isfinite(heights).any()
                if not args.model:
                    assert np.allclose(heights, 1.0, atol=1e-5)
                meta = json.loads(fetch(f"/api/result/{job}/meta.json"))
                assert meta["classes"]["1"] == "ground"
                assert "ndsm_std_m.npy" in status["files"]
                print("Inference smoke passed: authenticated API, tiled prediction, class metadata, uncertainty and result downloads.")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
