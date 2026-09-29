#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <chrono>

#define CUDA_CHECK(call) do {     cudaError_t err = (call);     if (err != cudaSuccess) {         std::cerr << "CUDA error: " << cudaGetErrorString(err)                   << " at " << __FILE__ << ":" << __LINE__ << std::endl;         return 1;     } } while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

static bool readPGM(const std::string& path, Image& img) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::string magic;
    in >> magic;
    if (magic != "P5") return false;

    auto skipComments = [&]() {
        while (in.peek() == '#') {
            std::string line;
            std::getline(in, line);
        }
    };

    skipComments();
    in >> img.width;
    skipComments();
    in >> img.height;
    skipComments();

    int maxValue;
    in >> maxValue;
    in.get();

    if (img.width <= 0 || img.height <= 0 || maxValue != 255) return false;

    img.pixels.resize(static_cast<size_t>(img.width) * img.height);
    in.read(reinterpret_cast<char*>(img.pixels.data()), img.pixels.size());
    return in.good();
}

static bool writePGM(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out << "P5\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.pixels.data()), img.pixels.size());
    return out.good();
}

__global__ void boxBlurKernel(const unsigned char* input,
                              unsigned char* output,
                              int width,
                              int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sum = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                ++count;
            }
        }
    }

    output[y * width + x] = static_cast<unsigned char>(sum / count);
}

static std::vector<std::string> listPGMFiles(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) return files;

    while (dirent* entry = readdir(dp)) {
        std::string name = entry->d_name;
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".pgm") {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(dp);

    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_directory> <output_directory>\n";
        return 1;
    }

    const std::string inputDir = argv[1];
    const std::string outputDir = argv[2];

    mkdir(outputDir.c_str(), 0755);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "CUDA GPU: " << prop.name << "\n";
    std::cout << "CUDA capability: " << prop.major << "." << prop.minor << "\n";

    std::vector<std::string> files = listPGMFiles(inputDir);
    if (files.empty()) {
        std::cerr << "No PGM images found in " << inputDir << "\n";
        return 1;
    }

    std::cout << "Images found: " << files.size() << "\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    size_t totalPixels = 0;
    float totalGpuMs = 0.0f;

    for (size_t i = 0; i < files.size(); ++i) {
        Image input;
        if (!readPGM(files[i], input)) {
            std::cerr << "Skipping unreadable image: " << files[i] << "\n";
            continue;
        }

        Image output;
        output.width = input.width;
        output.height = input.height;
        output.pixels.resize(input.pixels.size());

        unsigned char* d_input = nullptr;
        unsigned char* d_output = nullptr;
        size_t bytes = input.pixels.size();

        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.pixels.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((input.width + block.x - 1) / block.x,
                  (input.height + block.y - 1) / block.y);

        CUDA_CHECK(cudaEventRecord(start));
        boxBlurKernel<<<grid, block>>>(d_input, d_output, input.width, input.height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpuMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
        totalGpuMs += gpuMs;

        CUDA_CHECK(cudaMemcpy(output.pixels.data(), d_output, bytes, cudaMemcpyDeviceToHost));

        std::string filename = files[i].substr(files[i].find_last_of('/') + 1);
        std::string outPath = outputDir + "/blurred_" + filename;
        if (!writePGM(outPath, output)) {
            std::cerr << "Failed to write " << outPath << "\n";
        }

        totalPixels += bytes;
        cudaFree(d_input);
        cudaFree(d_output);

        if ((i + 1) % 20 == 0 || i + 1 == files.size()) {
            std::cout << "Processed " << (i + 1) << "/" << files.size()
                      << " images\n";
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "GPU kernel time (sum): " << totalGpuMs << " ms\n";
    std::cout << "Total pixels processed: " << totalPixels << "\n";
    std::cout << "Batch processing completed successfully.\n";

    return 0;
}
