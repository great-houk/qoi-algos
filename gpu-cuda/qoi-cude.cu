// We're importing all needed modules including the header file that contains the class definitions
#include "qoi-cude.hpp"
// This is the CUDA runtime API for memory management function like cudeMalloc, cudaMemcpy, cudaFree
#include <cuda_runtime.h>
// This includes definitions for CUDA Kernel launch parameters like threadIdx, blockIdx, blockDim
#include <device_launch_parameters.h>
// Standard library 
#include <cstdlib>
// For memory manipulation functions like memcpy and memcmp
#include <iostream>
// For I/O operations
#include <vector>

// Defining a macro that wraps CUDA API Calls to check for errors 
// We're executing the CUDA function call and we're storing the returned error code
// We're checking if the CUDA call failed (having any erorr code other than cudeSuccess)
#define CUDA_CHECK(call) \
    do { \
        cudeError_t err = call; \
        if(err != cudaSuccess) { \
            fprint(stderr, "Cuda Error at %s:%d - %s\n",__FILE__,__LINE__, \
            cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef union {
    struct {
        uint8_t r, g, b, a;
    } rgba;
    uint32_t v;
} qoi_rgba_t;
typedef union {
    struct {
        int32_t byte_offset;
        int32_t px_pos;
    } fields;
    uint8_t v[8];
} qoi_checkpoint_t;
static const uint8_t qoi_padding[8] = {0, 0, 0, 0, 0, 0, 0, 1};

_global__ void encode_segment_kernel(
    const uint8_t* pixels,
    uint8_t* d_output_segments,
    int32_t* d_segment_sizes,
    int segment_idx,
    int start_px,
    int num_px,
    int channels,
    int total_px,
    int max_segment_size
) {
    if(threadIdx.x != 0) return;
    // We're declaring a shared memory array of 64 pixels. 
    // We need this to be shared since shared memory is visible 
    // to all threads in a block and faster than global memory
    __shared__ qoi_rgba_t index[64];
    // We're initializing all 64 caches slots to INVALID_PIXEL
    for(int i = 0; i < 64; i++) {
        index[i].v = INVALID_PIXEL;
    }
    qoi_rgba_t px_prev;
    px_prev.v = INVALID_PIXEL;
    qoi_rgba_t px;
    px.rgba.r = 0;
    px.rgba.g = 0;
    px.rgba.b = 0;
    px.rgba.a = 255;
    int run = 0;
    int out_pos = 0;
    uint8_t* segment_out = d_output_segments + segment_idx * max_segment_size;
}