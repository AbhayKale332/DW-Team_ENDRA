"""Run with the target platform's Python after installing shared/requirements.txt."""
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
hidden = ["config", "serve.app", "dwdata.preprocess", "dwdata.scene_io",
          "infer.predict", "infer.engine", "geo.calibrate", "geo.dem",
          "viz.mesh", "viz.shadow", "models.tta", "uvicorn.logging", "uvicorn.loops.asyncio",
          "uvicorn.protocols.http.h11_impl", "uvicorn.lifespan.on"]
command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
           "--name", "depthwizard-inference", "--paths", str(root / "Model_Traning" / "v5"),
           "--distpath", str(here / "resources" / "backend"),
           "--workpath", str(here / ".build"), "--specpath", str(here / ".build"),
           "--collect-all", "onnxruntime", "--collect-all", "onnx",
           "--collect-all", "rasterio", "--collect-all", "pyproj"]
for module in hidden:
    command += ["--hidden-import", module]
for module in ["transformers", "huggingface_hub", "matplotlib", "pandas", "sklearn",
               "h5py", "cv2", "tensorboard", "pytest", "models.encoder", "models.heads", "train"]:
    command += ["--exclude-module", module]
command += [str(here / "inference.py")]
subprocess.run(command, cwd=here, check=True)
