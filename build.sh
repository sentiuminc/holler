#!/bin/bash
set -euo pipefail

DERIVED=".build/xcode"
PRODUCT_DIR="$DERIVED/Build/Products/Release"

if [[ "${1:-}" == "--clean" ]]; then
    echo "[holler] Cleaning build state..."
    rm -rf .build .swiftpm
fi

echo "[holler] Building release..."
xcodebuild -scheme holler \
    -configuration Release \
    -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED" \
    -quiet

cp "$PRODUCT_DIR/holler" ./holler
cp -R "$PRODUCT_DIR/mlx-swift_Cmlx.bundle" ./mlx-swift_Cmlx.bundle

echo "[holler] Done. Run: ./holler --text 'Hello world' --talk"
