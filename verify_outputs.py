#!/usr/bin/env python3
from pathlib import Path
import sys

EXPECTED_IMAGES = 200
EXPECTED_WIDTH = 256
EXPECTED_HEIGHT = 256
EXPECTED_PIXELS = EXPECTED_WIDTH * EXPECTED_HEIGHT

def read_pgm(path: Path):
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        dimensions = handle.readline().split()
        max_value = handle.readline().strip()
        data = handle.read()
    if magic != b"P5":
        raise ValueError("invalid PGM magic")
    if len(dimensions) != 2:
        raise ValueError("invalid PGM dimensions")
    width, height = map(int, dimensions)
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        raise ValueError(f"unexpected dimensions: {width}x{height}")
    if max_value != b"255":
        raise ValueError("unexpected maximum pixel value")
    if len(data) != EXPECTED_PIXELS:
        raise ValueError(f"unexpected pixel payload: {len(data)} bytes")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_directory>")
        return 1
    output_directory = Path(sys.argv[1])
    files = sorted(output_directory.glob("*.pgm"))
    if len(files) != EXPECTED_IMAGES:
        print(f"ERROR: expected {EXPECTED_IMAGES} PGM files, found {len(files)}")
        return 1
    for path in files:
        try:
            read_pgm(path)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {path}: {exc}")
            return 1
    print(f"Verified {len(files)} valid PGM outputs at {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
