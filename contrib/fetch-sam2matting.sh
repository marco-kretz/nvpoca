#!/bin/sh
# Fetch the SAM2Matting model code for --quality into nvpoca/sam2m. It is CC BY-NC 4.0, which the
# GPL-3.0 of this repo can't include, so it is downloaded instead of committed. The edits below make
# it importable as nvpoca.sam2m without SAM2's extra dependencies (hydra, iopath).
set -eu
COMMIT=73dd721d77b56749248aefe5e8824d7f61b9d13c
DEST=$(dirname "$0")/../nvpoca/sam2m

rm -rf "$DEST"
mkdir -p "$DEST"
curl -sSfL "https://codeload.github.com/FudanCVL/SAM2Matting/tar.gz/$COMMIT" \
  | tar -xz -C "$DEST" --strip-components=3 "SAM2Matting-$COMMIT/sam2/modeling"
curl -sSfL -o "$DEST/LICENSE" "https://raw.githubusercontent.com/FudanCVL/SAM2Matting/$COMMIT/LICENSE"

find "$DEST" -name '*.py' -exec sed -i 's/sam2\.modeling\./nvpoca.sam2m./g' {} +
sed -i -e '/^from iopath/d' -e 's/g_pathmgr\.open(/open(/' "$DEST/backbones/hieradet.py"
# only used by training-time point/box samplers
sed -i '/^from sam2\.utils\.misc import mask_to_box/d' "$DEST/sam2_utils.py"
# a float mask in the bf16 graph breaks the TensorRT export
sed -i 's/(unknown_region_inputs\.sigmoid() > 0\.65)\.float()/(unknown_region_inputs.sigmoid() > 0.65).to(mask_inputs.dtype)/' \
  "$DEST/sam2matting_base.py"
grep -q 'to(mask_inputs.dtype)' "$DEST/sam2matting_base.py" || { echo "dtype patch did not apply" >&2; exit 1; }
echo "SAM2Matting $COMMIT -> $DEST"
