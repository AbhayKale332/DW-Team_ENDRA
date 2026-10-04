"""Run with the target platform's Python after installing shared/requirements.txt."""
from pathlib import Path
import subprocess
import shutil
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
           # The ONNX imports and runtime hook collect required code and provider libraries.
           # collect-all also bundles model-zoo fixtures, exporters and tooling.
           "--collect-all", "rasterio", "--collect-all", "pyproj"]
for module in hidden:
    command += ["--hidden-import", module]
for module in ["transformers", "huggingface_hub", "matplotlib", "pandas", "sklearn",
               "h5py", "cv2", "tensorboard", "pytest", "models.encoder", "models.heads", "train"]:
    command += ["--exclude-module", module]
command += [str(here / "inference.py")]
subprocess.run(command, cwd=here, check=True)

# PyInstaller's torch hook copies native test programs as package data.
# Keep the actual tensor libraries and shared-memory helper.
torch_dir = here / "resources" / "backend" / "depthwizard-inference" / "_internal" / "torch"
shutil.rmtree(torch_dir / "test", ignore_errors=True)
for binary in (torch_dir / "bin").glob("*"):
    name = binary.stem.lower()
    if binary.is_file() and (name.startswith("test_") or name.endswith("test") or name.startswith("tutorial_")):
        binary.unlink()

# Keep dynamic exports and loadable code; discard unused native symbol tables on Linux.
# GNU strip is supplied by binutils (preinstalled on the Linux release runner).
if sys.platform == "linux":
    for library in torch_dir.parent.rglob("*.so*"):
        if library.is_file() and not library.is_symlink():
            subprocess.run(["strip", "--strip-unneeded", str(library)], check=True)
