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

__global__ void encode_segment_kernel(
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

    for(int px_idx = 0; px_idx < num_px; px_idx++) {
        int global_px_pos = (start_px * px_idx) * channels;
        // loads current pixel RGB
        px.rgba.r = pixels[global_px_pos + 0];
		px.rgba.g = pixels[global_px_pos + 1];
		px.rgba.b = pixels[global_px_pos + 2];
		// loads alpha of current pixel if exists
		if (channels == 4) {
			px.rgba.a = pixels[global_px_pos + 3];
		}
		// If current pixel is same as previous pixel
		if (px.v == px_prev.v) {
			run++;
			if (run == 62) {
				segment_out[out_pos++] = QOI_OP_RUN | (run - 1);
				run = 0;
			}
		} else {
			// if pixel color changed we push to byte and flush run
			if (run > 0) {
				segment_out[out_pos++] = QOI_OP_RUN | (run - 1);
				run = 0;
			}

			// Hash current pixel to a color index slot [0..63].
			// basically maping the pixel to a position in the index table
			int index_pos = QOI_COLOR_HASH(px) & (64 - 1);

			// Fast path: if the hash slot holds the exact same pixel, emit
			// INDEX op.
			if (index[index_pos].v != INVALID_PIXEL &&
				index[index_pos].v == px.v) {
				segment_out[out_pos++] = QOI_OP_INDEX | index_pos;
			} else {
				// Otherwise, update the index slot with this pixel.
				index[index_pos] = px;

				// If px_prev was invalidated (e.g., at start or after a
				// checkpoint), we must write this pixel absolutely
				// (RGB/RGBA).
				if (px_prev.v == INVALID_PIXEL) {
					if (channels == 4) {
						segment_out[out_pos++] = QOI_OP_RGBA;
					} else {
						segment_out[out_pos++] = QOI_OP_RGB;
					}

					segment_out[out_pos++] = px.rgba.r;
					segment_out[out_pos++] = px.rgba.g;
					segment_out[out_pos++] = px.rgba.b;
					if (channels == 4) {
						segment_out[out_pos++] = px.rgba.a;
					}
				} else {
					// If alpha is unchanged, try DIFF/LUMA (smaller) before
					// falling back to RGB.
					if (px.rgba.a == px_prev.rgba.a) {
						signed char vr = px.rgba.r - px_prev.rgba.r;
						signed char vg = px.rgba.g - px_prev.rgba.g;
						signed char vb = px.rgba.b - px_prev.rgba.b;

						signed char vg_r = vr - vg;
						signed char vg_b = vb - vg;

						if (vr > -3 && vr < 2 && vg > -3 && vg < 2 && vb > -3 &&
							vb < 2) {
							segment_out[out_pos++] = QOI_OP_DIFF | (vr + 2) << 4 |
											(vg + 2) << 2 | (vb + 2);
						} else if (vg_r > -9 && vg_r < 8 && vg > -33 &&
								   vg < 32 && vg_b > -9 && vg_b < 8) {
							segment_out[out_pos++] = QOI_OP_LUMA | (vg + 32);
							segment_out[out_pos++] = (vg_r + 8) << 4 | (vg_b + 8);
						} else {
							segment_out[out_pos++] = QOI_OP_RGB;
							segment_out[out_pos++] = px.rgba.r;
							segment_out[out_pos++] = px.rgba.g;
							segment_out[out_pos++] = px.rgba.b;
						}
					} else {
						segment_out[out_pos++] = QOI_OP_RGBA;
						segment_out[out_pos++] = px.rgba.r;
						segment_out[out_pos++] = px.rgba.g;
						segment_out[out_pos++] = px.rgba.b;
						segment_out[out_pos++] = px.rgba.a;
					}
				}
			}
		}
		px_prev = px;
	}
	// Flush any remaining run.
	if (run > 0) {
		segment_out[out_pos++] = QOI_OP_RUN | (run - 1);
	}

    d_segment_sizes[segment_idx] = out_pos;

}

__global__ void decode_segment_kernel(
    const uint8_t* d_encoded,
    uint8_t* d_pixels,
    qoi_checkpoint_t checkpoint,
    int next_px_pos,
    int channels,
    int chuncks_len
) {

    if(threadIdx.x != 0) return;

    // Initalize per checkpoint state
    __shared__ qoi_rgba_t index[64];
    for(int i = 0; i < 64; i++) {
        index[i].v = 0;
    }

    qoi_rgba_t px;
    px.rgba.r = 0;
    px.rgba.g = 0;
    px.rgba.b = 0;
    px.rgba.a = 255;

    int run = 0;
    // Set per checkpoint variables
    int p = checkpoint.fields.byte_offset;
    int px_pos = checkpoint.fields.px_pos;

    while(px_pos < next_px_pos) {
        if(run>0) {
            run--;
        } else if (p < chuncks_len) {
            uint8_t b1 = d_encoded[p++];
            if(b1 == QOI_OP_RGB) {
                px.rgba.r = d_encoded[p++];
                px.rgba.g = d_encoded[p++];
                px.rgba.b = d_encoded[p++];
            }
            else if(b1 == QOI_OP_RGBA) {
                px.rgba.r = d_encoded[p++];
                px.rgba.g = d_encoded[p++];
                px.rgba.b = d_encoded[p++];
                px.rgba.a = d_encoded[p++];

            }
            else if((b1 & QOI_MASK_2) == QOI_OP_INDEX) {
                px = index[b1 & 0x3F];
            }
            else if((b1 & QOI_MASK_2) == QOI_OP_DIFF) {
                px.rgba.r += ((b1 >> 4) & 0x03) - 2;
                px.rgba.g += ((b1 >> 4) & 0x03) - 2;
                px.rgba.b += (b1 & 0x03) - 2;
            }
            else if((b1 & QOI_MASK_2) == QOI_OP_LUMA) {
                uint8_t b2 = d_encoded[p++];
                int vg = (b1 & 0x3F) - 32;
                px.rgba.r += vg - 8 + ((b2 >> 4) & 0x0F);
                px.rgba.g += vg;
                px.rgba.b += vg - 8 + (b2 & 0x0F);
            }
            else if((b1 & QOI_MASK_2) == QOI_OP_RUN) {
                run = (b1 & 0x3F);
            }
            index[QOI_COLOR_HASH(px) & 63] = px;

        }
        d_pixels[px_pos + 0] = px.rgba.r;
        d_pixels[px_pos + 1] = px.rgba.g;
        d_pixels[px_pos + 2] = px.rgba.b;
        if(channels == 4) {
            d_pixels[px_pos + 3] = px.rgba.a;
        }
        px_pos += channels;
    }
}