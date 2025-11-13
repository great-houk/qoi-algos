CXX      ?= g++
# Makefile for bench
# The compiler can be overridden by the environment (e.g. bench.slurm exports CXX=g++-14)
CXXFLAGS := -std=c++2a -g -O3 -Wall -Wextra -I. -fopenmp
LDFLAGS  := -lstdc++ -lm -lpng
TARGET   := bench

SRCS := bench.cpp test_params.cpp convert.cpp reference/qoi-reference.cpp reference/libpng.cpp reference/stb_image.cpp single-cpu/qoi-sc.cpp multi-cpu/qoi-mc.cpp
OBJS := $(patsubst %.cpp, bin/%.o, $(SRCS))

TESTPARAM_OBJS := $(filter-out bin/convert.o bin/bench.o, $(OBJS))
CONVERT_OBJS := $(filter-out bin/bench.o bin/test_params.o, $(OBJS))
BENCH_OBJS := $(filter-out bin/convert.o bin/test_params.o, $(OBJS))

.PHONY: all clean run

all: bin/$(TARGET) clean_errors

.PHONY: convert
convert: bin/convert
	./bin/convert
# 	rm -rf images/*.png
# 	rm -rf images_small/*.png

bin/$(TARGET): $(BENCH_OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(BENCH_OBJS) -o bin/$(TARGET) $(LDFLAGS)


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

run:
	./bin/$(TARGET) 1 images

clean_errors:
	rm -rf images/*.err.qoi
	rm -rf images_small/*.err.qoi

clean:
	rm -rf bin