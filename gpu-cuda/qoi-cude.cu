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