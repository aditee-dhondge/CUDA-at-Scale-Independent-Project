#!/bin/bash
set -euo pipefail

INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
EXPECTED_IMAGES=200

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

echo "Generating ${EXPECTED_IMAGES} deterministic grayscale test images..."
python3 - <<'PY'
from pathlib import Path

output = Path('data/input')
output.mkdir(parents=True, exist_ok=True)
width = height = 256
count = 200

for index in range(count):
    path = output / f'image_{index:03d}.pgm'
    with path.open('wb') as handle:
        handle.write(f'P5\n{width} {height}\n255\n'.encode())
        pixels = bytearray(width * height)
        for y in range(height):
            for x in range(width):
                pixels[y * width + x] = (x + y + index * 7 + ((x * y) % 31)) % 256
        handle.write(pixels)

print(f'Generated {count} images of {width}x{height} pixels.')
PY

echo "Building CUDA program..."
make clean
make

echo "Running GPU batch image processing..."
rm -f "$OUTPUT_DIR"/*.pgm
./bin/cuda_batch_image_blur "$INPUT_DIR" "$OUTPUT_DIR"

output_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.pgm' | wc -l | tr -d ' ')
echo "Output images: ${output_count}"

if [ "$output_count" -ne "$EXPECTED_IMAGES" ]; then
    echo "ERROR: Expected ${EXPECTED_IMAGES} output images, but found ${output_count}."
    exit 1
fi

echo "Verifying output image files..."
python3 verify_outputs.py "$OUTPUT_DIR"

echo "GPU batch image processing completed successfully for all ${EXPECTED_IMAGES} images."
