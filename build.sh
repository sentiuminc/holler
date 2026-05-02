#!/bin/bash
set -euo pipefail

DERIVED=".build/xcode"
PRODUCT_DIR="$DERIVED/Build/Products/Release"

# Clean stale SPM state that can confuse xcodebuild
rm -rf .build .swiftpm

echo "[holler] Building release (this takes ~3 min on first run)..."
xcodebuild -scheme holler \
    -configuration Release \
    -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED" \
    -quiet

# Copy binary + Metal shaders to repo root so ./holler just works
cp "$PRODUCT_DIR/holler" ./holler
cp -R "$PRODUCT_DIR/mlx-swift_Cmlx.bundle" ./mlx-swift_Cmlx.bundle

echo "[holler] Done. Run: ./holler --text \"Hello world\" --talk"
