# CUDA at Scale Independent Project

## Overview

This project demonstrates GPU-accelerated image processing at scale using CUDA.

The program generates a deterministic dataset of **200 grayscale images**, each 256x256 pixels, and applies a 3x3 box-blur filter to every image using a custom CUDA kernel. The input images are generated locally so the repository does not need to store a large binary dataset.

The project demonstrates:

- CUDA GPU computation with a custom CUDA kernel
- Batch processing of 200 images
- Host-to-device and device-to-host memory transfers
- CUDA event timing for GPU kernel execution
- Automated dataset generation and execution

## Code Organization

- `src/batch_image_blur.cu` - CUDA image-processing program
- `Makefile` - builds the CUDA executable
- `run.sh` - generates 200 images, builds the program, and processes the complete batch
- `data/input/` - generated input images
- `data/output/` - generated blurred images
- `bin/` - compiled executable

## How to Run

A CUDA-capable Linux environment with `nvcc` is required.

```bash
chmod +x run.sh
./run.sh
```

The execution prints the GPU name, CUDA capability, number of images, progress, total GPU kernel time, and total pixels processed.

## GPU Processing

The image filter is implemented in `boxBlurKernel` and executed with a 2D CUDA grid using 16x16 threads per block.

For every output pixel, the kernel averages the valid pixels in its 3x3 neighborhood.

## Expected Proof

A successful run should report:

```
CUDA GPU: <GPU name>
Images found: 200
Processed 200/200 images
GPU kernel time (sum): <time> ms
Total pixels processed: 13107200
Batch processing completed successfully.
Output images: 200
```

The exact GPU name and timing depend on the laboratory GPU.
