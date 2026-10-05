## Set up the v5 code

Use Python with a compatible PyTorch/torchvision installation for your CPU or CUDA environment. The v5 requirements deliberately leave PyTorch unpinned.

```bash
git clone https://github.com/AbhayKale332/DW-Team_ENDRA.git
cd DepthWizard/Model_Traning/v5
pip install torch torchvision
pip install -r requirements.txt
pip install kagglehub
```

The model loader currently initialises DINOv3 from Hugging Face before loading the fine-tuned state. Accept the backbone's access terms and authenticate your environment for `facebook/dinov3-vitl16-pretrain-sat493m` if required:

```bash
hf auth login
```

Initial loading needs network access or an already populated Hugging Face cache. Downloading this Kaggle checkpoint alone does not remove that loader dependency.

## Download the checkpoint

```python
from pathlib import Path
import kagglehub

model_dir = Path(kagglehub.model_download(
    "abhaydkale232/depthwizard-v5/pyTorch/forest-final/1"
))
print(model_dir / "best.pt")
```

Use the printed absolute path as `--ckpt` below. On Kaggle, you can instead attach the model through **Add Input** and locate `best.pt` under the mounted input directory. Allow approximately 2.5 GB for the checkpoint, plus room for libraries, the backbone cache and generated outputs. Authenticate to Kaggle and ensure your account has access when downloading a private model.

## Scene inference

Run these commands from `DepthWizard/Model_Traning/v5`. Replace `/absolute/path/to/best.pt` and the image path with your files.

```bash
# Georeferenced RGB GeoTIFF: GSD is read from the scene metadata.
python -m infer.predict scene.tif \
    --ckpt /absolute/path/to/best.pt \
    --out-dir outputs/scene --device cuda --no-mesh

# PNG/JPEG: supply the measured GSD in metres per pixel.
python -m infer.predict scene.png \
    --ckpt /absolute/path/to/best.pt --gsd 0.5 \
    --out-dir outputs/scene --device cpu --no-mesh

# Print the checkpoint's preprocessing contract without scene prediction.
python -m infer.predict scene.tif \
    --ckpt /absolute/path/to/best.pt --report
```

Set `--device cpu` when CUDA is unavailable; CPU inference can be slow for this ViT-L model. Reduce `--batch-tiles` to 1 if memory is constrained. Optional `--tta` enables eight-way dihedral test-time augmentation and increases inference work substantially; the published validation results do not establish a TTA accuracy gain for this checkpoint.

Typical products include `ndsm_m.npy` (metres above ground), `ndsm16.png` (encoded preview), `rgb.png`, segmentation products, uncertainty products when available and `ndsm.tif` for georeferenced inputs. Read the generated product metadata for file names, encoding, units, datum and any calibration errors. Large-scene windowed inference may provide overview and full-resolution products separately.

## Optional absolute DSM

```bash
# Fetch the default elevation reference for a georeferenced scene.
python -m infer.predict scene.tif \
    --ckpt /absolute/path/to/best.pt --absolute \
    --out-dir outputs/absolute --device cuda --no-mesh

# Or use a local DEM with its known vertical datum.
python -m infer.predict scene.tif \
    --ckpt /absolute/path/to/best.pt \
    --dem terrain.tif --dem-datum EGM2008 \
    --out-dir outputs/absolute --device cuda --no-mesh
```

Specify the actual datum of your local DEM; `EGM2008` here is an example. The default `dem_anchored` mode preserves coarse-cell means of the reference while adding finer detail. Check DEM coverage and output metadata: without usable coverage, the pipeline may return nDSM without an absolute DSM. An unreferenced PNG/JPEG cannot establish geographic position or an absolute elevation datum by itself.

## Direct Python loading

For model integration, use the project's loader so architecture and preprocessing come from the checkpoint:

```python
import torch
from infer.predict import load_model

model, spec, cfg = load_model(
    "/absolute/path/to/best.pt", torch.device("cpu")
)

# rgb_tile must already be a valid, stretched RGB uint8 tile at
# spec.canonical_gsd_m, with shape (spec.tile_size, spec.tile_size, 3).
x = torch.from_numpy(spec.normalise(rgb_tile)).unsqueeze(0)
with torch.inference_mode():
    prediction = model(x)

height_m = prediction["fused"][0, 0].cpu().numpy()
class_ids = prediction["seg"].argmax(1)[0].cpu().numpy()
bin_spread_m = prediction["b_std"][0, 0].cpu().numpy()
```

`rgb_tile` represents your prepared image; `spec.normalise` applies only normalisation, not stretching or GSD resampling. Use `infer.predict` for full scenes so geospatial reading, scale handling, tiling and blending remain consistent. Bin spread is not a calibrated error interval.

## Optional ONNX export

This release does not include an ONNX model. Export it with the v5 code after the checkpoint and backbone can be loaded:

```bash
python -m infer.export_onnx \
    --ckpt /absolute/path/to/best.pt \
    --out outputs/onnx/depthwizard.onnx --opset 18
```

Export verification compares PyTorch and ONNX height outputs when ONNX Runtime is installed. Inspect the verification result before deploying. Ship the graph **together with** `depthwizard.onnx.json` and every external weight sidecar named in that metadata. The graph takes normalised RGB at a fixed spatial tile size with a dynamic batch axis and exposes `height_m`, `seg` and `height_std_m`. Once exported, graph execution can use ONNX Runtime without downloading DINOv3.
