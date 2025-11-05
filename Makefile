# Makefile for bench

CXX      := g++
CXXFLAGS := -std=c++17 -g -O3 -Wall -Wextra -I.
LDFLAGS  := -lstdc++ -lm -lpng
TARGET   := bench

SRCS := bench.cpp reference/qoi-reference.cpp reference/libpng.cpp reference/stb_image.cpp single-cpu/qoi-sc.cpp
OBJS := $(patsubst %.cpp, bin/%.o, $(SRCS))

.PHONY: all clean run

all: bin/$(TARGET)

bin/$(TARGET): $(OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJS) -o bin/$(TARGET) $(LDFLAGS)

bin/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run:
	./bin/$(TARGET) 10 images

clean:
	rm -rf bin