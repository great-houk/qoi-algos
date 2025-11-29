CXX      ?= g++
NVCC     ?= nvcc
# Makefile for bench
# The compiler can be overridden by the environment (e.g. bench.slurm exports CXX=g++-14)
CXXFLAGS := -std=c++2a -g -O3 -Wall -Wextra -I. -fopenmp
NVCCFLAGS := -std=c++17 -O3 -I. --compiler-options -fPIC
LDFLAGS  := -lstdc++ -lm -lpng
CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart
TARGET   := bench

SRCS := bench.cpp test_params.cpp convert.cpp reference/qoi-reference.cpp reference/libpng.cpp reference/stb_image.cpp single-cpu/qoi-sc.cpp multi-cpu/qoi-mc.cpp gpu/qoi-gpu.cpp gpu/qoi-gpu.hpp
# CUDA_SRCS := gpu-cuda/qoi-cuda.cu
OBJS := $(patsubst %.cpp, bin/%.o, $(SRCS))
# CUDA_OBJS := $(patsubst %.cu, bin/%.o, $(CUDA_SRCS))

TESTPARAM_OBJS := $(filter-out bin/convert.o bin/bench.o, $(OBJS))
CONVERT_OBJS := $(filter-out bin/bench.o bin/test_params.o, $(OBJS))
BENCH_OBJS := $(filter-out bin/convert.o bin/test_params.o, $(OBJS)) $(CUDA_OBJS)

.PHONY: all clean run

all: bin/$(TARGET) clean_errors

.PHONY: convert
convert: bin/convert
	./bin/convert
# 	rm -rf images/*.png
# 	rm -rf images_small/*.png

bin/$(TARGET): $(BENCH_OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(BENCH_OBJS) -o bin/$(TARGET) $(LDFLAGS) $(CUDA_LDFLAGS)


bin/test_params: $(TESTPARAM_OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(TESTPARAM_OBJS) -o bin/test_params $(LDFLAGS)

.PHONY: test_params
test_params: bin/test_params
	@echo "Running parameter sweep (default: 3 runs, images_small -> test_params.csv)"
	./bin/test_params 3 images_small test_params.csv


bin/convert: $(CONVERT_OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(CONVERT_OBJS) -o bin/convert $(LDFLAGS)

bin/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run:
	./bin/$(TARGET) 1 images

clean_errors:
	rm -rf images/*.err.qoi
	rm -rf images_small/*.err.qoi

clean:
	rm -rf bin