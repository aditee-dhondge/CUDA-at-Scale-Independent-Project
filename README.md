# CUDA Batch Image Blur Processing

A GPU-accelerated image-processing capstone project using NVIDIA CUDA. The application generates a deterministic batch of grayscale images and applies a 3x3 box-blur filter to every image using a custom CUDA kernel.

## Project Goal

The project demonstrates how a repetitive image-processing workload can be moved from the CPU to the GPU and executed in parallel across many pixels and many images.

Key techniques:
- Custom CUDA kernel development
- Two-dimensional CUDA grids and 16x16 thread blocks
- Host-to-device and device-to-host memory transfers
- CUDA event timing
- Batch processing of 200 images
- Command-line input/output directories
- Automated dataset generation
- Output validation
- Reproducible build and execution

## Problem and Motivation

A 3x3 blur requires each output pixel to inspect a small neighborhood of input pixels. An image contains thousands or millions of pixels, so the same calculation is repeated many times. CUDA is appropriate because independent pixel calculations can be assigned to many GPU threads.

The project uses generated grayscale PGM images so the complete demonstration is deterministic and does not depend on an external dataset.

## Processing Pipeline

```text
Generate 200 deterministic PGM images
            |
            v
       Input directory
            |
            v
     Read image on CPU
            |
            v
   Allocate GPU memory
            |
            v
   Copy CPU -> GPU
            |
            v
   Launch BoxBlurKernel
     16 x 16 threads
            |
            v
   Copy GPU -> CPU
            |
            v
    Write blurred PGM
            |
            v
   Validate 200 outputs
```

## CUDA Kernel

The `BoxBlurKernel` assigns one CUDA thread to each output pixel. Each thread computes its two-dimensional coordinate, visits neighboring offsets from -1 through +1, checks image boundaries, averages valid grayscale values, and writes the result.

## GPU Memory and Timing

Input data is loaded into host memory. Device memory is allocated with `cudaMalloc`, and pixels are transferred with `cudaMemcpy`. CUDA events surround each kernel launch, and the application accumulates kernel execution time over the batch.

The timing represents GPU kernel execution, not complete end-to-end application time, because file I/O, allocation, and other CPU-side work are outside the measured kernel interval.

## Command-Line Interface

```bash
./bin/cuda_batch_image_blur <input_directory> <output_directory>
```

## Repository Structure

```text
CUDA-at-Scale-Independent-Project/
├── src/batch_image_blur.cu
├── docs/project_description.md
├── docs/presentation_script.md
├── proof_of_execution.txt
├── verify_outputs.py
├── Makefile
├── run.sh
├── INSTALL
├── README.md
└── .gitignore
```

## Requirements

- Linux environment
- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- Python 3
- GNU Make

The documented execution used an NVIDIA L4 GPU with compute capability 8.9.

## Build

```bash
make
```

## Reproduce the Complete Demonstration

```bash
chmod +x run.sh
./run.sh
```

The validation stage checks that exactly 200 output files exist, each is a binary PGM, each image is 256x256, the maximum value is 255, and each image contains 65,536 pixel bytes.

## Recorded Execution

- GPU: NVIDIA L4
- CUDA compute capability: 8.9
- Input images: 200
- Image dimensions: 256x256
- Total pixels: 13,107,200
- Output images: 200
- Summed GPU kernel time: 2.15501 ms

The recorded execution completed the complete batch successfully.

## Results and Interpretation

The execution demonstrates that the CUDA kernel can process a large number of independent pixel calculations as one repeatable batch. The measured 2.15501 ms value is the accumulated GPU kernel time from the documented run.

It should not be interpreted as total application latency because CPU file reading/writing, allocation, and transfer overhead are outside the measured kernel interval.

## Design Decisions

### Deterministic input generation
The project generates its own input images so another reviewer can reproduce the workload without downloading a dataset.

### 16x16 CUDA blocks
A 16x16 block provides 256 threads and maps naturally to two-dimensional image coordinates.

### Boundary handling
The kernel checks neighboring coordinates before reading them. This avoids invalid global-memory accesses and gives edge pixels an average based only on valid neighbors.

### Batch processing
The application processes every PGM image found in the input directory rather than hard-coding a single image.

### Output validation
The run script performs a second validation stage after CUDA execution. This provides evidence that the expected number and format of output files were produced.

## Challenges

1. Mapping image coordinates to CUDA blocks and threads.
2. Handling edge pixels without out-of-bounds accesses.
3. Managing device allocation and host/device transfers.
4. Measuring kernel execution separately from CPU file-processing overhead.
5. Creating a reproducible batch workflow.
6. Validating generated outputs automatically.

## Lessons Learned

The project reinforced that GPU acceleration requires more than writing a kernel. Memory movement, allocation strategy, workload decomposition, and measurement methodology all influence application performance.

The two-dimensional structure of image data also makes CUDA's grid/block model intuitive: blocks cover regions of the image while individual threads operate on individual pixels.

## Future Improvements

- Use CUDA streams to overlap data transfers and computation.
- Use shared-memory tiling to reuse neighboring pixels.
- Keep image data resident on the GPU across multiple processing stages.
- Add a CPU reference implementation for direct benchmarking.
- Add Gaussian blur and Sobel edge detection.
- Support color images and larger image formats.
- Compare different block dimensions experimentally.

## Capstone Materials

- Project description: `docs/project_description.md`
- Presentation script: `docs/presentation_script.md`
- Execution evidence: `proof_of_execution.txt`
- Output validation: `verify_outputs.py`
