NVCC ?= nvcc
TARGET := bin/cuda_batch_image_blur
SRC := src/batch_image_blur.cu

CXXFLAGS := -O2
GENCODE ?= -arch=sm_70

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p bin
	$(NVCC) $(CXXFLAGS) $(GENCODE) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
