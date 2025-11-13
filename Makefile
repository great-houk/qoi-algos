# Makefile for bench

CXX      := g++
CXXFLAGS := -std=c++2a -g -O3 -Wall -Wextra -I. -fopenmp
LDFLAGS  := -lstdc++ -lm -lpng
TARGET   := bench

SRCS := bench.cpp reference/qoi-reference.cpp reference/libpng.cpp reference/stb_image.cpp single-cpu/qoi-sc.cpp multi-cpu/qoi-mc.cpp
OBJS := $(patsubst %.cpp, bin/%.o, $(SRCS))

.PHONY: all clean run

all: bin/$(TARGET) clean_errors

bin/$(TARGET): $(OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/$(TARGET) $(LDFLAGS)

bin/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run:
	./bin/$(TARGET) 1 images

clean_errors:
	rm -rf images/*.err.png
	rm -rf images_small/*.err.png

clean:
	rm -rf bin