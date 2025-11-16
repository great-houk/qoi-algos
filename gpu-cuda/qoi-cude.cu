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