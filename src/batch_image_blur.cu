#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

static bool ReadPgm(const std::string& path, Image& image) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;

    std::string magic;
    input >> magic;
    if (magic != "P5") return false;

    auto SkipComments = [&input]() {
        input >> std::ws;
        while (input.peek() == '#') {
            std::string line;
            std::getline(input, line);
            input >> std::ws;
        }
    };

    SkipComments();
    input >> image.width;
    SkipComments();
    input >> image.height;
    SkipComments();

    int max_value = 0;
    input >> max_value;
    input.get();

    if (image.width <= 0 || image.height <= 0 || max_value != 255) {
        return false;
    }

    const size_t pixel_count =
        static_cast<size_t>(image.width) * image.height;
    image.pixels.resize(pixel_count);
    input.read(reinterpret_cast<char*>(image.pixels.data()), pixel_count);

    return input.good();
}

static bool WritePgm(const std::string& path, const Image& image) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;

    output << "P5\n" << image.width << " " << image.height << "\n255\n";
    output.write(reinterpret_cast<const char*>(image.pixels.data()),
                 image.pixels.size());
    return output.good();
}

__global__ void BoxBlurKernel(const unsigned char* input,
                              unsigned char* output,
                              int width,
                              int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sum = 0;
    int count = 0;

    for (int delta_y = -1; delta_y <= 1; ++delta_y) {
        for (int delta_x = -1; delta_x <= 1; ++delta_x) {
            const int neighbor_x = x + delta_x;
            const int neighbor_y = y + delta_y;

            if (neighbor_x >= 0 && neighbor_x < width &&
                neighbor_y >= 0 && neighbor_y < height) {
                sum += input[neighbor_y * width + neighbor_x];
                ++count;
            }
        }
    }

    output[y * width + x] =
        static_cast<unsigned char>(sum / count);
}

static std::vector<std::string> ListPgmFiles(const std::string& directory) {
    std::vector<std::string> files;
    DIR* directory_handle = opendir(directory.c_str());
    if (directory_handle == nullptr) return files;

    while (dirent* entry = readdir(directory_handle)) {
        const std::string name = entry->d_name;
        if (name.size() >= 4 &&
            name.compare(name.size() - 4, 4, ".pgm") == 0) {
            files.push_back(directory + "/" + name);
        }
    }

    closedir(directory_handle);
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_directory> <output_directory>\n";
        return 1;
    }

    const std::string input_directory = argv[1];
    const std::string output_directory = argv[2];
    mkdir(output_directory.c_str(), 0755);

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

    std::cout << "CUDA GPU: " << device_properties.name << "\n";
    std::cout << "CUDA capability: " << device_properties.major << "."
              << device_properties.minor << "\n";

    const std::vector<std::string> files = ListPgmFiles(input_directory);
    if (files.empty()) {
        std::cerr << "No PGM images found in " << input_directory << "\n";
        return 1;
    }

    std::cout << "Images found: " << files.size() << "\n";

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    size_t total_pixels = 0;
    size_t processed_images = 0;
    float total_gpu_ms = 0.0f;

    for (size_t index = 0; index < files.size(); ++index) {
        Image input;
        if (!ReadPgm(files[index], input)) {
            std::cerr << "ERROR: Unable to read " << files[index] << "\n";
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            return 1;
        }

        Image output;
        output.width = input.width;
        output.height = input.height;
        output.pixels.resize(input.pixels.size());

        unsigned char* device_input = nullptr;
        unsigned char* device_output = nullptr;
        const size_t bytes = input.pixels.size();

        CUDA_CHECK(cudaMalloc(&device_input, bytes));
        CUDA_CHECK(cudaMalloc(&device_output, bytes));
        CUDA_CHECK(cudaMemcpy(device_input, input.pixels.data(), bytes,
                              cudaMemcpyHostToDevice));

        const dim3 block(16, 16);
        const dim3 grid((input.width + block.x - 1) / block.x,
                        (input.height + block.y - 1) / block.y);

        CUDA_CHECK(cudaEventRecord(start_event));
        BoxBlurKernel<<<grid, block>>>(device_input, device_output,
                                       input.width, input.height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
        total_gpu_ms += gpu_ms;

        CUDA_CHECK(cudaMemcpy(output.pixels.data(), device_output, bytes,
                              cudaMemcpyDeviceToHost));

        const size_t slash = files[index].find_last_of('/');
        const std::string filename = files[index].substr(slash + 1);
        const std::string output_path =
            output_directory + "/blurred_" + filename;

        if (!WritePgm(output_path, output)) {
            std::cerr << "ERROR: Failed to write " << output_path << "\n";
            cudaFree(device_input);
            cudaFree(device_output);
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            return 1;
        }

        total_pixels += input.pixels.size();
        ++processed_images;

        cudaFree(device_input);
        cudaFree(device_output);

        if (processed_images % 20 == 0 ||
            processed_images == files.size()) {
            std::cout << "Processed " << processed_images << "/"
                      << files.size() << " images\n";
        }
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    std::cout << "GPU kernel time (sum): " << total_gpu_ms << " ms\n";
    std::cout << "Total pixels processed: " << total_pixels << "\n";
    std::cout << "Batch processing completed successfully.\n";

    return 0;
}
