# CUDA Batch Image Blur Processing

A GPU-accelerated image-processing capstone project using NVIDIA CUDA. The application generates a deterministic batch of grayscale images and applies a 3x3 box-blur filter to every image using a custom CUDA kernel.

## Project Goal

The goal is to demonstrate how a repetitive image-processing workload can be moved from the CPU to the GPU and executed in parallel across many pixels and many images.

- Custom CUDA kernel development
- 2D CUDA grids and 16x16 thread blocks
- Host-to-device and device-to-host memory transfers
- CUDA event timing
- Batch processing of 200 images
- Command-line input/output directories
- Automated data generation and reproducible execution
- Verification of the produced output batch

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
   Copy image CPU -> GPU
            |
            v
     Launch boxBlurKernel
       16 x 16 threads
            |
            v
   Copy result GPU -> CPU
            |
            v
    Write blurred PGM
            |
            v
      Output directory
```

## CUDA Kernel

The boxBlurKernel assigns one CUDA thread to each output pixel. Each thread examines the valid pixels in its 3x3 neighborhood, computes their average, and writes the result to the output image.

## Repository Structure

```text
CUDA-at-Scale-Independent-Project/
├── src/
│   └── batch_image_blur.cu
├── docs/
│   ├── project_description.md
│   └── presentation_script.md
├── proof_of_execution.txt
├── Makefile
├── run.sh
├── INSTALL
├── README.md
└── .gitignore
```

## Requirements

- Linux environment
- NVIDIA GPU
- CUDA Toolkit with nvcc
- Python 3
- GNU Make

The project was executed successfully on an NVIDIA L4 GPU with CUDA compute capability 8.9.

## Build

```bash
make
```

The executable is created at bin/cuda_batch_image_blur.

## Run Manually

The program accepts two command-line arguments:

```bash
./bin/cuda_batch_image_blur <input_directory> <output_directory>
```

Example:

```bash
./bin/cuda_batch_image_blur data/input data/output
```

This provides a clear command-line interface for the capstone project.

## Reproduce the Complete Demonstration

The supplied script generates the deterministic input dataset, builds the CUDA application, runs the complete batch, and verifies that all 200 output images were produced.

```bash
chmod +x run.sh
./run.sh
```

Expected final messages include:

```text
CUDA GPU: NVIDIA L4
CUDA capability: 8.9
Images found: 200
Processed 200/200 images
GPU kernel time (sum): <measured time> ms
Total pixels processed: 13107200
Batch processing completed successfully.
Output images: 200
```

Exact timing varies with the GPU and execution environment.

## Execution Evidence

A recorded execution from the course laboratory is stored in proof_of_execution.txt. The recorded run processed 200 images at 256x256 pixels, for 13,107,200 total pixels, on an NVIDIA L4 GPU. It produced 200 output images and measured 2.15501 ms of summed GPU kernel time.

## Performance Measurement

CUDA events are placed around each kernel launch. The elapsed GPU kernel times are accumulated across the batch. The reported value is the sum of kernel execution times and does not include all CPU-side file I/O.

## Design Decisions

### Deterministic input generation
The project generates its own input images instead of requiring a large external dataset. This makes the demonstration reproducible and keeps the repository lightweight.

### 16x16 CUDA blocks
A 16x16 block provides 256 threads per block and maps naturally to the two-dimensional structure of an image.

### Boundary handling
Pixels near an image edge have fewer than nine valid neighbors. The kernel checks image boundaries and averages only valid neighbors.

### Batch processing
The application processes every PGM file found in the input directory, making the workload easy to scale to a larger number of images.

## Challenges and Lessons Learned

1. Mapping a two-dimensional image to CUDA thread and block coordinates.
2. Handling image boundaries correctly without reading outside allocated memory.
3. Managing GPU memory allocation and data transfers safely.
4. Measuring GPU kernel execution separately from file-processing overhead.
5. Building a reproducible CUDA workflow that can process a complete batch automatically.

## Future Improvements

- Keep image data resident on the GPU across multiple processing stages.
- Process multiple images concurrently using CUDA streams.
- Use shared memory or tiled loading to reduce repeated global-memory reads.
- Compare GPU performance with a CPU implementation.
- Add additional filters such as Sobel edge detection and Gaussian blur.
- Add support for larger image formats and color images.

## Capstone Materials

- Project description: docs/project_description.md
- Presentation script: docs/presentation_script.md
- Execution proof: proof_of_execution.txt

## License

This project is provided for educational and course-capstone purposes.