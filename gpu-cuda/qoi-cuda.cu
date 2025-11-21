// We're importing all needed modules including the header file that contains the class definitions
#include "qoi-cuda.hpp"
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
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",__FILE__,__LINE__, \
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
    int channels,
    int total_px,
    int max_segment_size,
    int seg_px,
    int num_segments
) {

    int segment_idx = threadIdx.x +  blockIdx.x * threadIdx.x;

    // We're declaring a shared memory array of 64 pixels. 
    // We need this to be shared since shared memory is visible 
    // to all threads in a block and faster than global memory
    qoi_rgba_t index[64];
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

    int start_px = segment_idx * seg_px;
    int end_px = (segment_idx == num_segments - 1) ? total_px : start_px + seg_px;
    int num_px = end_px - start_px;

    for(int px_idx = 0; px_idx < num_px; px_idx++) {
        int global_px_pos = (start_px + px_idx) * channels; 
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
    const qoi_checkpoint_t* d_checkpoints,
    const int* d_next_px_pos,
    int channels,
    int chuncks_len
) {

    int cp_idx = threadIdx.x +  blockIdx.x * threadIdx.x;
    qoi_checkpoint_t checkpoint = d_checkpoints[cp_idx];
    int next_px_pos = d_next_px_pos[cp_idx];

    // Initalize per checkpoint state
    qoi_rgba_t index[64];
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
                px.rgba.g += ((b1 >> 2) & 0x03) - 2;
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

std::vector<uint8_t> CUDAQOI::encode(const std::vector<uint8_t>& pixels,
    QOIEncoderSpec& spec) {
    if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}
	int channels = spec.channels;
	int total_px = spec.width * spec.height;

	int num_segments =
		(total_px * PIXEL_TO_ENCODE_RATIO) / this->CHECKPOINT_INTERVAL + 1;
	// int checks_per_seg = this->CHECKPOINTS_PER_SEGMENT;
	int seg_px = total_px / num_segments;

    std::vector<qoi_checkpoint_t> checkpoints(num_segments);
    uint8_t* d_pixels;
    uint8_t* d_output_segments;
    int32_t* d_segment_sizes;

    size_t pixel_bytes = total_px * channels;
    int max_segment_size = seg_px * (channels + 1);
    size_t total_output_size = num_segments * max_segment_size;

    CUDA_CHECK(cudaMalloc(&d_pixels, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_segments, total_output_size));
    CUDA_CHECK(cudaMalloc(&d_segment_sizes, num_segments * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_pixels, pixels.data(), pixel_bytes, cudaMemcpyHostToDevice));
    int threads_per_block = num_segments;
    int blocks = 1;

    encode_segment_kernel<<<blocks,threads_per_block>>>(
        d_pixels, d_output_segments, d_segment_sizes,
        channels, total_px, max_segment_size, seg_px, num_segments
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> segment_size(num_segments);
    CUDA_CHECK(cudaMemcpy(segment_size.data(), d_segment_sizes, num_segments * sizeof(int32_t), cudaMemcpyDeviceToHost));
    // Update checkpoints
    uint32_t data_nbytes = sizeof(QOIHeader);
    for (int s = 0; s < num_segments; s++) {
		checkpoints[s].fields.byte_offset = data_nbytes;
		checkpoints[s].fields.px_pos = s * seg_px * channels;
        data_nbytes += segment_size[s];
	}

    // Allocate output bytes
	uint32_t total_bytes = data_nbytes + sizeof(qoi_padding) +
						   num_segments * sizeof(qoi_checkpoint_t) +
						   sizeof(qoi_padding);
    std::vector<uint8_t>output(total_bytes);
   	QOIHeader header = {.fields = {
							.magic = QOI_MAGIC,
							.width = spec.width,
							.height = spec.height,
							.channels = spec.channels,
							.colorspace = spec.colorspace,
						}};
    // QOI Uses big endian for some reason :(
	header.fields.magic = __builtin_bswap32(header.fields.magic);
	header.fields.width = __builtin_bswap32(header.fields.width);
	header.fields.height = __builtin_bswap32(header.fields.height);
	memcpy(output.data(), header.b, sizeof(QOIHeader));
    for(int s = 0; s < num_segments; s++) {
        int offset = checkpoints[s].fields.byte_offset;
        int size = segment_size[s];
        CUDA_CHECK(cudaMemcpy(
            output.data() + offset,
            d_output_segments + s * max_segment_size,
            size,
            cudaMemcpyDeviceToHost
        ));
    }

    // Append footer, checkpoints, and second footer
	memcpy(output.data() + data_nbytes, qoi_padding, sizeof(qoi_padding));
    int checkpoint_offset = data_nbytes + sizeof(qoi_padding);
	for (auto c : checkpoints) {
		memcpy(output.data() + checkpoint_offset, c.v, sizeof(c.v));
		checkpoint_offset += sizeof(c.v);
	}
	memcpy(output.data() + checkpoint_offset, qoi_padding, sizeof(qoi_padding));
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_output_segments));
    CUDA_CHECK(cudaFree(d_segment_sizes));
    return output;
}

std::vector<uint8_t> CUDAQOI::decode(
	const std::vector<uint8_t>& encoded_data,
	QOIDecoderSpec& spec) {

    QOIHeader header;
	memcpy(header.b, encoded_data.data(), sizeof(QOIHeader));

    uint32_t header_magic = __builtin_bswap32(header.fields.magic);
    spec.width = __builtin_bswap32(header.fields.width);
	spec.height = __builtin_bswap32(header.fields.height);
	spec.channels = header.fields.channels;
	spec.colorspace = header.fields.colorspace;

    if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 || header_magic != QOI_MAGIC ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}

    
	int channels = spec.channels;
	int px_len = spec.width * spec.height * channels;
	int max_size = spec.width * spec.height * (spec.channels + 1) +
				   sizeof(QOIHeader) + sizeof(qoi_padding);
	// Read in checkpoints
	typedef union {
		struct {
			int32_t byte_offset;
			int32_t px_pos;
			int32_t next_px_pos;
		} fields;
		// Only 8, since next_px_pos is not stored in file
		char v[8];
	} checkpoint_span_t;
    int max_num_checkpoints = (max_size / this->CHECKPOINT_INTERVAL) + 1;
    std::vector<checkpoint_span_t> checkpoints;
    bool found_double_padding = false;
    int last_cp_px_pos = px_len;
    int p_size = (int)sizeof(qoi_padding);
	int c_size = (int)sizeof(qoi_checkpoint_t);
    const uint8_t* cp_p = encoded_data.data() + encoded_data.size() - p_size;
    	for (int i = 0; i < max_num_checkpoints; i++) {
		// Check if we found the second padding
		uint8_t padding[8];
		memcpy(padding, cp_p - p_size, p_size);
		if (memcmp(padding, qoi_padding, p_size) == 0) {
			found_double_padding = true;
			break;
		}
		// Read in checkpoint
		checkpoint_span_t cp;
		memcpy(cp.v, cp_p - c_size, c_size);
		// We're reading backwards, so set next_px_pos from last read
		cp.fields.next_px_pos = last_cp_px_pos;
		last_cp_px_pos = cp.fields.px_pos;
		checkpoints.push_back(cp);
		// Move backwards to continue reading cps
		cp_p -= sizeof(qoi_checkpoint_t);
	}
	if (!found_double_padding) {
		// No double padding found, reset to no checkpoints
		checkpoints.clear();
		checkpoints.push_back(
			(checkpoint_span_t){.fields = {sizeof(QOIHeader), 0, px_len}});
		std::cout << "WARNING: No checkpoints found in QOI data, decoding from "
					 "beginning only."
				  << std::endl;
	}

    int num_checkpoints = (int)checkpoints.size();
    std::vector<qoi_checkpoint_t> h_checkpoints(num_checkpoints);
    std::vector<int> h_next_px_pos(num_checkpoints);
    for (int i = 0; i < num_checkpoints; i++) {
        h_checkpoints[i].fields.byte_offset = checkpoints[i].fields.byte_offset;
        h_checkpoints[i].fields.px_pos = checkpoints[i].fields.px_pos;
        h_next_px_pos[i] = checkpoints[i].fields.next_px_pos;
    }

    uint8_t* d_encoded;
    uint8_t* d_pixels;
    qoi_checkpoint_t* d_checkpoints;
    int* d_next_px_pos;
    CUDA_CHECK(cudaMalloc(&d_encoded, encoded_data.size()));
    CUDA_CHECK(cudaMalloc(&d_pixels, px_len));
    CUDA_CHECK(cudaMalloc(&d_checkpoints, num_checkpoints * sizeof(qoi_checkpoint_t)));
    CUDA_CHECK(cudaMalloc(&d_next_px_pos, num_checkpoints * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_encoded, encoded_data.data(), encoded_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_checkpoints, h_checkpoints.data(), num_checkpoints * sizeof(qoi_checkpoint_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_px_pos, h_next_px_pos.data(), num_checkpoints * sizeof(int), cudaMemcpyHostToDevice));
    int chuncks_len = encoded_data.size() - sizeof(qoi_padding);
    int threads_per_block = num_checkpoints;
    int blocks = 1;

    decode_segment_kernel <<<blocks, threads_per_block>>>(
        d_encoded, d_pixels, d_checkpoints, d_next_px_pos, channels, chuncks_len
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> pixels(px_len);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, px_len, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_encoded));
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_checkpoints));
    CUDA_CHECK(cudaFree(d_next_px_pos));
    return pixels;
}
