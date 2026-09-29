# Kaggle datasets (as of 2026-09-24)

What a DepthWizard Kaggle notebook sees under `/kaggle/input` (the `./` below).
The listing is a sample: five files per directory, not every file.

- **Mount root:** `/kaggle/input/datasets/abhaydkale232/<dataset>/`.
- **Packed stores** (`depthwizard-*`), uploaded from the Modal `depthwizard-data`
  Volume by `V4_modal/03_publish_kaggle.ipynb`. Each split is a memmap store:
  `index.json`, `stretch_bounds_2_98.npy` and `shard_NNN_{rgb,hgt,cls,val}.npy`.
  `train.py` reads these directly, with no packing step.
  - `gamus` and `india-labeled` mount one level deeper: `<split>/<split>/`. Each
    split was zipped with its own folder inside it.
  - `synrs3d-g05` and `synrs3d-g1` mount at `train/` directly.
  - `bash run_kaggle.sh link` (v5, DAV2_V1) symlinks both layouts into one
    `data_root/<source>/<split>/`, e.g. `depthwizard-india-labeled` becomes
    `india_labeled`.
- **Raw DFC23 Track 2** (`dfc23-track2-height-estimation`) is not packed.
  - Only `track2/train` has heights (`dsm/`). `track2/val` and
    `track2_test_data` hold only rgb and sar: the contest withheld those labels.
  - The top-level `rgb/`, `dsm/` and `sar/` are a copy of `track2/train`.
  - Pack it in the notebook: `python prepare_data.py --datasets dfc23 --dfc23_dir
    <root>/track2/train --dfc23_tile 512 --dfc23_max_height_m 150`. That writes
    `dfc23_g0XX/{train,val}`, with val split off by scene.
- **Not on Kaggle:**
  - a packed `dfc23_g050`
  - `india_unlabeled`
  - `geonrw`
  - the GAMUS PNG mirror (`akashch1512/gamusdataset`), which has been removed.

```
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/README.md
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/rgb
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/rgb/SV_Brasilia_-15.8466_-47.8966.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/rgb/SV_NewYork_40.7388_-74.0108.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/rgb/SV_SanDiego_32.7275_-117.2434.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/rgb/SV_Copenhagen_55.6770_12.5319.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/rgb/GF2_Brasilia_-15.8652_-47.9381.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/dsm
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/dsm/SV_Brasilia_-15.8466_-47.8966.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/dsm/SV_NewYork_40.7388_-74.0108.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/dsm/SV_SanDiego_32.7275_-117.2434.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/dsm/SV_Copenhagen_55.6770_12.5319.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/dsm/GF2_Brasilia_-15.8652_-47.9381.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/buildings_only_train.json
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/rgb
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/rgb/P_0199.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/rgb/P_0309.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/rgb/P_0090.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/rgb/P_0284.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/rgb/P_0239.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/sar
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/sar/P_0199.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/sar/P_0309.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/sar/P_0090.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/sar/P_0284.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/val/sar/P_0239.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/rgb
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/rgb/SV_Brasilia_-15.8466_-47.8966.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/rgb/SV_NewYork_40.7388_-74.0108.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/rgb/SV_SanDiego_32.7275_-117.2434.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/rgb/SV_Copenhagen_55.6770_12.5319.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/rgb/GF2_Brasilia_-15.8652_-47.9381.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/dsm
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/dsm/SV_Brasilia_-15.8466_-47.8966.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/dsm/SV_NewYork_40.7388_-74.0108.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/dsm/SV_SanDiego_32.7275_-117.2434.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/dsm/SV_Copenhagen_55.6770_12.5319.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/dsm/GF2_Brasilia_-15.8652_-47.9381.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/sar
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/sar/SV_Brasilia_-15.8466_-47.8966.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/sar/SV_NewYork_40.7388_-74.0108.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/sar/SV_SanDiego_32.7275_-117.2434.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/sar/SV_Copenhagen_55.6770_12.5319.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2/train/sar/GF2_Brasilia_-15.8652_-47.9381.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/sar
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/sar/SV_Brasilia_-15.8466_-47.8966.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/sar/SV_NewYork_40.7388_-74.0108.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/sar/SV_SanDiego_32.7275_-117.2434.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/sar/SV_Copenhagen_55.6770_12.5319.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/sar/GF2_Brasilia_-15.8652_-47.9381.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/rgb
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/rgb/P_0663.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/rgb/P_1107.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/rgb/P_0708.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/rgb/P_0635.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/rgb/P_0897.tif
📁 ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/sar
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/sar/P_0663.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/sar/P_1107.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/sar/P_0708.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/sar/P_0635.tif
  ├── ./datasets/abhaydkale232/dfc23-track2-height-estimation/track2_test_data/sar/P_0897.tif
📁 ./datasets/abhaydkale232/depthwizard-india-labeled/val/val
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/val/val/stretch_bounds_2_98.npy
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/val/val/shard_000_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/val/val/index.json
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/val/val/shard_000_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/val/val/shard_000_cls.npy
📁 ./datasets/abhaydkale232/depthwizard-india-labeled/train/train
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/train/train/stretch_bounds_2_98.npy
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/train/train/shard_000_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/train/train/index.json
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/train/train/shard_000_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-india-labeled/train/train/shard_000_cls.npy
📁 ./datasets/abhaydkale232/depthwizard-synrs3d-g1/train
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g1/train/shard_001_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g1/train/shard_004_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g1/train/shard_004_cls.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g1/train/shard_010_rgb.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g1/train/shard_009_hgt.npy
📁 ./datasets/abhaydkale232/depthwizard-synrs3d-g05/train
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g05/train/shard_001_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g05/train/shard_004_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g05/train/shard_004_cls.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g05/train/shard_011_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-synrs3d-g05/train/shard_010_rgb.npy
📁 ./datasets/abhaydkale232/depthwizard-gamus/val/val
  ├── ./datasets/abhaydkale232/depthwizard-gamus/val/val/shard_001_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/val/val/shard_004_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/val/val/shard_004_cls.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/val/val/shard_003_rgb.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/val/val/shard_004_rgb.npy
📁 ./datasets/abhaydkale232/depthwizard-gamus/test/test
  ├── ./datasets/abhaydkale232/depthwizard-gamus/test/test/shard_001_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/test/test/shard_015_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/test/test/shard_015_rgb.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/test/test/shard_004_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/test/test/shard_004_cls.npy
📁 ./datasets/abhaydkale232/depthwizard-gamus/train/train
  ├── ./datasets/abhaydkale232/depthwizard-gamus/train/train/shard_039_rgb.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/train/train/shard_035_rgb.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/train/train/shard_001_hgt.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/train/train/shard_015_val.npy
  ├── ./datasets/abhaydkale232/depthwizard-gamus/train/train/shard_023_val.npy
```
