# Makefile for bench

CXX      := g++
CXXFLAGS := -std=c++11 -g -O3 -Wall -Wextra -I. -I/usr/include/stb
LDFLAGS  := -lstdc++ -lm
TARGET   := bench

SRCS := bench.cpp qoi-reference.cpp single-cpu/qoi-sc.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)