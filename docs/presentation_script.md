# 5–10 Minute Capstone Presentation Script

## Slide 1 — Title (30 seconds)

**CUDA Batch Image Blur Processing**

My project is CUDA Batch Image Blur Processing. The goal was to demonstrate how a repetitive image-processing workload can be accelerated using NVIDIA CUDA. The application generates a reproducible batch of 200 grayscale images and applies a 3x3 box blur using a custom CUDA kernel.

## Slide 2 — Problem and Motivation (45 seconds)

Image filtering is naturally parallel because the calculation for one output pixel is mostly independent of other output pixels. CUDA allows individual pixels to be assigned to GPU threads so many calculations can execute concurrently.

I selected image blurring because it is simple to explain while still demonstrating grids, blocks, global memory, kernel execution, and timing.

## Slide 3 — Project Workflow (45 seconds)

The workflow has five stages: generate 200 deterministic images, read an image on the CPU, copy pixels to GPU memory, execute the CUDA blur kernel, and copy and save the result. The run script automates the complete process.

## Slide 4 — CUDA Kernel (1 minute)

The main CUDA component is BoxBlurKernel. Each thread calculates its x and y pixel coordinates using CUDA block and thread indices. It examines a 3x3 neighborhood, adds valid grayscale values, and divides by the number of valid neighbors. Boundary checks handle pixels at the edges.

## Slide 5 — Memory and Execution (1 minute)

The application uses cudaMalloc for device memory and cudaMemcpy for host-to-device and device-to-host transfers. CUDA events measure kernel execution time for each image. This demonstrates that GPU programming involves memory management and timing as well as kernel code.

## Slide 6 — Batch Processing and Results (1 minute)

The application processes 200 images, each 256 by 256 pixels, which is 13,107,200 pixels in total. The documented run used an NVIDIA L4 with compute capability 8.9 and produced 200 blurred output images. The summed GPU kernel time was 2.15501 milliseconds.

## Slide 7 — Challenges and Lessons Learned (1 minute)

The main challenges were mapping two-dimensional image coordinates to CUDA threads, handling image boundaries safely, managing device memory, and separating GPU kernel timing from CPU file-processing overhead.

I also learned how to create a reproducible CUDA application that processes a complete batch instead of only a single test image.

## Slide 8 — Future Improvements (45 seconds)

Future work could use CUDA streams to overlap transfers and computation, shared-memory tiling to reduce repeated global-memory reads, GPU-resident multi-stage processing, CPU-versus-GPU benchmarking, and additional filters such as Gaussian blur and Sobel edge detection.

## Slide 9 — Demonstration and Conclusion (1 minute)

For the demonstration, I would show the repository, the command-line interface, and the execution output.

```bash
./bin/cuda_batch_image_blur data/input data/output
```

The output reports the GPU, image count, processing progress, kernel timing, total pixels, and generated output count.

Overall, this project helped me apply CUDA concepts to a complete GPU-based application combining CUDA programming, memory management, GPU timing, batch processing, and reproducible execution.
