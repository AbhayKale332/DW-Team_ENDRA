#!/bin/bash
set -e

# ---- CONFIG: edit these ----
KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
DATASET_SLUG="dfc23-track2-height-estimation"
DATASET_TITLE="DFC23 Track2 Building Height Estimation"
ROOT_DIR="$HOME"                          # project root on the Lightning.ai studio
STAGE_DIR="$HOME/kaggle_upload/DFC23"     # clean staging folder built for upload
# -----------------------------

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# Copy only the processed data relevant to single-view height estimation.
# Skips: raw zips (data/DFC23, track2_Test.zip), IDE/shell/cache junk (.idea, .vscode,
# .cache, .config, .ipython, .zshrc, .zcompdump, .lightning_studio, .git).
cp -r "$ROOT_DIR/track2"            "$STAGE_DIR/track2"
cp -r "$ROOT_DIR/track2_test_data"  "$STAGE_DIR/track2_test_data"
cp -r "$ROOT_DIR/rgb"               "$STAGE_DIR/rgb"
cp -r "$ROOT_DIR/sar"               "$STAGE_DIR/sar"
cp -r "$ROOT_DIR/dsm"               "$STAGE_DIR/dsm"

# track2/train.zip is already a zip archive of the actual training data -
# unzip it so the dataset browses cleanly on Kaggle instead of nesting a zip inside a zip.
if [ -f "$STAGE_DIR/track2/train.zip" ]; then
  unzip -q "$STAGE_DIR/track2/train.zip" -d "$STAGE_DIR/track2/train"
  rm "$STAGE_DIR/track2/train.zip"
fi

echo "Staged dataset size:"
du -sh "$STAGE_DIR"

# ---- Kaggle metadata ----
cat > "$STAGE_DIR/dataset-metadata.json" << EOF
{
  "title": "$DATASET_TITLE",
  "id": "$KAGGLE_USERNAME/$DATASET_SLUG",
  "licenses": [{"name": "other"}]
}
EOF

# ---- README with required DFC23 attribution ----
cat > "$STAGE_DIR/README.md" << 'EOF'
# DFC23 Track 2 - Building Extraction & Height Estimation

Subset of the 2023 IEEE GRSS Data Fusion Contest (DFC23) Track 2 data,
used for single-view height estimation (DepthWizard project).

Contents:
- track2/train        - training optical (rgb) + SAR images, building annotations, reference nDSMs
- track2/val           - validation rgb/sar images
- track2_test_data     - test rgb/sar images (no reference)
- rgb, sar, dsm        - curated sample triplets (Portsmouth, New York x2, New Delhi)

## Citation
"[REF. NO.] 2023 IEEE GRSS Data Fusion Contest. Online:
www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/"

## Acknowledgement
The authors would like to thank the IEEE GRSS Image Analysis and Data Fusion
Technical Committee, Aerospace Information Research Institute, Chinese Academy
of Sciences, Universitat der Bundeswehr Munchen, and GEOVIS Earth Technology
Co., Ltd. for organizing the Data Fusion Contest.

## Reference paper
Huang, X., Ren, L., Liu, C., Wang, Y., Yu, H., Schmitt, M., Hansch, R., Sun, X.,
Huang, H., Mayer, H., 2022. Urban Building Classification (UBC) - A Dataset for
Individual Building Detection and Classification from Satellite Imagery.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 1413-1421.
EOF

# ---- Create (first time) ----
# --dir-mode zip is required for the CLI to include subfolders at all (sar, dsm,
# track2, track2_test_data, rgb). Kaggle unpacks the zip server-side, so the
# dataset page still shows browsable individual files, not a raw .zip download.
kaggle datasets create -p "$STAGE_DIR" --dir-mode zip

# For subsequent updates, comment the line above and use instead:
# kaggle datasets version -p "$STAGE_DIR" -m "Updated data" --dir-mode zip
