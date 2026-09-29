# Project Description

## Problem

Image filtering is a highly parallel workload because each output pixel can be calculated independently from neighboring input pixels. This project implements a grayscale 3x3 box blur as a CUDA kernel and applies it to a batch of 200 images.

## Approach

1. Generate deterministic 256x256 grayscale PGM images.
2. Read each image on the CPU.
3. Allocate GPU memory and copy the image from host to device.
4. Launch a CUDA grid using 16x16 threads per block.
5. Assign one CUDA thread to each output pixel.
6. Average the valid pixels in the 3x3 neighborhood.
7. Copy the result from device to host.
8. Write the blurred image as a PGM file.

## CUDA Mapping

For each output pixel, the kernel calculates its x and y coordinates from block and thread indices. The thread then visits offsets from -1 through +1 in both dimensions. Boundary checks prevent invalid memory accesses.

## GPU Concepts Demonstrated

- Parallel execution with CUDA threads
- Two-dimensional grids and blocks
- cudaMalloc and cudaMemcpy
- Device global memory
- Host/device data transfers
- CUDA event timing
- Batch GPU processing

## Results

The documented laboratory execution used an NVIDIA L4 GPU with compute capability 8.9. It processed 200 images of 256x256 pixels, for 13,107,200 total pixels, and produced 200 output images. The recorded summed GPU kernel time was 2.15501 ms.

## Challenges

- Correctly mapping two-dimensional image coordinates to CUDA threads.
- Handling image boundaries without out-of-bounds memory accesses.
- Managing GPU allocations and memory transfers.
- Separating kernel timing from CPU-side file I/O.
- Creating a reproducible workflow for an entire image batch.

## Lessons Learned

The project demonstrates that GPU acceleration is most useful when a workload contains many independent operations. It also shows that kernel execution time is only one part of an application: file I/O, memory allocation, and host-device transfers also affect end-to-end performance.

## Future Improvements

- Use CUDA streams to overlap data transfers and computation.
- Use shared-memory tiling to reuse neighboring pixels.
- Keep data resident on the GPU across multiple image-processing stages.
- Compare against a CPU implementation.
- Add Gaussian blur and Sobel edge detection.
- Support color images and larger image formats.
