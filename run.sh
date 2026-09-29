#!/bin/bash
set -e

mkdir -p data/input data/output

echo "Generating 200 deterministic grayscale test images..."
python3 - <<'PY'
import os
from pathlib import Path

out = Path("data/input")
out.mkdir(parents=True, exist_ok=True)

width = height = 256
count = 200

for i in range(count):
    path = out / f"image_{i:03d}.pgm"
    with path.open("wb") as f:
        f.write(f"P5\\n{width} {height}\\n255\\n".encode())
        data = bytearray(width * height)
        for y in range(height):
            for x in range(width):
                value = (x + y + i * 7 + ((x * y) % 31)) % 256
                data[y * width + x] = value
        f.write(data)

print(f"Generated {count} images of {width}x{height} pixels.")
PY

echo "Building CUDA program..."
make

echo "Running GPU batch image processing..."
rm -f data/output/*.pgm
./bin/cuda_batch_image_blur data/input data/output

echo "Output images: $(find data/output -name '*.pgm' | wc -l)"
