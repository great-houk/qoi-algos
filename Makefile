# Makefile for bench

CXX      := g++
CXXFLAGS := -std=c++2a -g -O3 -Wall -Wextra -I. -fopenmp
LDFLAGS  := -lstdc++ -lm -lpng
TARGET   := bench

SRCS := bench.cpp convert.cpp reference/qoi-reference.cpp reference/libpng.cpp reference/stb_image.cpp single-cpu/qoi-sc.cpp multi-cpu/qoi-mc.cpp
OBJS := $(patsubst %.cpp, bin/%.o, $(SRCS))

CONVERT_OBJS := $(filter-out bin/bench.o, $(OBJS))
BENCH_OBJS := $(filter-out bin/convert.o, $(OBJS))

.PHONY: all clean run

all: bin/$(TARGET) clean_errors

.PHONY: convert
convert: bin/convert
	./bin/convert
	rm -rf images/*.png
	rm -rf images_small/*.png

bin/$(TARGET): $(BENCH_OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(BENCH_OBJS) -o bin/$(TARGET) $(LDFLAGS)


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