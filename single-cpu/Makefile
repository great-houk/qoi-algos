# Makefile for bench.c
# Builds the benchmark program using libpng and standard libraries

CC      := gcc
CFLAGS  := -std=gnu99 -O3 -Wall -Wextra $(shell pkg-config --cflags libpng)
LDFLAGS := $(shell pkg-config --libs libpng) -lm
TARGET  := bench
SRC     := bench.c qoi_sc_wrapper.c qoi_ref_wrapper.c 

# Optional headers (stb_image, stb_image_write, qoi)
# must be in the same directory or include path
INCLUDES := -I.

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) $(LDFLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) 10 ../images
	

clean:
	rm -f $(TARGET)
