#!/bin/bash
set -e

DATA_DIR="/teamspace/studios/this_studio/data/DFC23"
mkdir -p "$DATA_DIR"

FILES=(
  "DFC23_IEEE-DataPort.zip"
  "track1.zip"
  "track2.zip"
  "roof_fine_train_corrected.json"
  "track1_test_data.zip"
  "track2_test_data.zip"
)

for f in "${FILES[@]}"; do
  echo "Downloading $f ..."
  aws s3 cp "s3://ieee-dataport/competition/1137541/$f" "$DATA_DIR/$f"
done

echo "Download complete."
du -sh "$DATA_DIR"/*
