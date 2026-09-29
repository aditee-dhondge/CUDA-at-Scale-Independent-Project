NVCC ?= nvcc
TARGET := bin/cuda_batch_image_blur
SRC := src/batch_image_blur.cu

NVCCFLAGS := -O2 -std=c++14

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) data/input data/output

clean:
	rm -rf bin