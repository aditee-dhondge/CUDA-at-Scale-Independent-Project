# Project Description

## Purpose

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA. The workload is a 3x3 grayscale box blur applied to a reproducible batch of 200 images. The goal is to show how a repetitive, data-parallel operation can be mapped onto CUDA threads and measured as part of a complete application.

## Algorithm

Each output pixel is assigned to one CUDA thread. The thread computes its two-dimensional coordinate from CUDA block and thread indices, examines the surrounding 3x3 neighborhood, ignores neighbors outside the image boundary, averages the valid grayscale values, and writes the result.

The program uses 16x16 CUDA thread blocks, giving 256 threads per block and a natural mapping for two-dimensional image data.

## Memory and Execution

The input image is loaded into host memory. Device memory is allocated with cudaMalloc and the image is copied from host to device with cudaMemcpy. BoxBlurKernel processes the image in parallel. The result is copied back to host memory and written as a PGM file.

CUDA events are used around each kernel launch. The application accumulates kernel execution time over the batch. This is deliberately treated separately from file I/O and other CPU-side overhead so that GPU measurement is not confused with end-to-end latency.

## Batch and Validation

The demonstration uses 200 deterministic 256x256 grayscale images. The complete batch contains 13,107,200 pixels. The run script creates the inputs, builds the CUDA executable, runs GPU processing, checks that 200 output files were generated, and validates their PGM format and dimensions.

This makes the experiment reproducible and provides evidence that the application operates on a batch rather than only one sample.

## Results

The documented execution used an NVIDIA L4 GPU with compute capability 8.9. It processed all 200 images and generated 200 output images. The recorded summed GPU kernel time was 2.15501 ms.

The timing is the accumulated CUDA kernel time for the batch; it is not total application runtime because CPU file I/O, allocation, and transfer overhead are outside the measured kernel interval.

## Challenges

The main implementation challenges were mapping two-dimensional image coordinates to CUDA threads, handling boundary pixels safely, managing device memory and transfers, and designing a repeatable batch workflow.

Another challenge was interpreting performance correctly. A GPU kernel can be very fast while the overall program is still affected by disk I/O and memory transfers. Separating these components makes the performance result more meaningful.

## Lessons Learned

The project showed how CUDA's grid and block model maps naturally to image data. It also reinforced that GPU performance depends on the complete data path: host memory, device memory, transfers, kernel execution, and output processing.

The project demonstrated the value of reproducibility. Generating deterministic input data and validating every output makes it easier for another reviewer to repeat the experiment and verify that the application behaved as intended.

## Future Work

The next improvements would be to keep data resident on the GPU across multiple image-processing stages, use shared-memory tiling to reduce repeated global-memory reads, overlap transfers and computation with CUDA streams, and add a CPU reference implementation for a direct end-to-end comparison. Additional filters such as Gaussian blur and Sobel edge detection could then be added as further GPU kernels.
