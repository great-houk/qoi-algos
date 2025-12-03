#include <stdbool.h>
#include <vector>
#include <iostream>
#include "vector.hpp"


#include "qoi-gpu.hpp"

typedef union {
	struct {
		uint8_t r, g, b, a;
	} rgba;
	uint32_t v;
} qoi_rgba_t;

static const uint8_t qoi_padding[8] = {0, 0, 0, 0, 0, 0, 0, 1};
#define segment_bit 12
#define segment ((1 << segment_bit) - 1)

std::vector<uint8_t> GPUQOI::encode(const std::vector<uint8_t>& pixels,
									QOIEncoderSpec& spec) {
	// Allocate
	std::vector<uint8_t> data;
	// Each pixel takes ~1.8 bytes on avg to encode
	data.reserve(spec.width * spec.height * 2);

	// Write header
	data.resize(sizeof(QOIHeader));
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
	memcpy(&data[0], header.b, sizeof(QOIHeader));

	// Write pixels
	int padding_needed = 0;
	int last_seg_p = 0;
#define B (data.size() - sizeof(QOIHeader))

	int err_cnt = 0;
	bool attempted = false;

	int p = 0;

	qoi_rgba_t index[64];
	qoi_rgba_t px_prev;
	qoi_rgba_t px = {.v = 0xFF'00'00'00};  // Full alpha black

	while (p < pixels.size()) {
		// Segment logic
		if ((B & segment) == 0) {
			memset(index, 0, sizeof(index));
			px_prev.v = INVALID_PIXEL;
			last_seg_p = p;
		}
		int last_b = B;

		// Load Pixel
		memcpy(&px.v, &pixels[p], 3);
		p += spec.channels;

		// Calculate differences
#define TWOB(x) (x >= -2 && x <= 1)
#define FOURB(x) (x >= -8 && x <= 7)
#define SIXB(x) (x >= -32 && x <= 31)
		int8_t vr = px.rgba.r - px_prev.rgba.r;
		int8_t vg = px.rgba.g - px_prev.rgba.g;
		int8_t vb = px.rgba.b - px_prev.rgba.b;

		int8_t vg_r = vr - vg;
		int8_t vg_b = vb - vg;

		// Calculate hash
		uint8_t index_pos = QOI_COLOR_HASH(px) & (64 - 1);

		// Invalid previous
		if (px_prev.v == INVALID_PIXEL) {
			if (padding_needed == 0) {
				data.push_back(QOI_OP_RGB);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
			} else {
				data.push_back(QOI_OP_RGBA);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				data.push_back(255);
				padding_needed--;
			}
		}
		// Run
		else if (px_prev.v == px.v) {
			if (padding_needed < 3) {
				int run = 0;
				qoi_rgba_t px_next = px;
				while (run < 61 && p < pixels.size()) {
					memcpy(&px_next.v, &pixels[p], 3);
					if (px.v != px_next.v) {
						break;
					}

					p += spec.channels;
					run++;
				}
				data.push_back(QOI_OP_RUN | run);
			} else if (padding_needed == 3) {
				data.push_back(QOI_OP_RGB);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				padding_needed -= 3;
			} else {
				data.push_back(QOI_OP_RGBA);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				data.push_back(255);
				padding_needed -= 4;
			}
		}
		// Hash
		else if (index[index_pos].v == px.v) {
			if (padding_needed < 3) {
				data.push_back(QOI_OP_INDEX | index_pos);
			} else if (padding_needed == 3) {
				data.push_back(QOI_OP_RGB);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				padding_needed -= 3;
			} else {
				data.push_back(QOI_OP_RGBA);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				data.push_back(255);
				padding_needed -= 4;
			}
		}
		// Diff
		else if (TWOB(vr) && TWOB(vg) && TWOB(vb)) {
			if (padding_needed == 0) {
				uint8_t byte =
					QOI_OP_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2);
				data.push_back(byte);
			} else if (padding_needed < 3) {
				data.push_back(QOI_OP_LUMA | (vg + 32));
				data.push_back((vg_r + 8) << 4 | (vg_b + 8));
				padding_needed--;
			} else if (padding_needed == 3) {
				data.push_back(QOI_OP_RGB);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				padding_needed -= 3;
			} else {
				data.push_back(QOI_OP_RGBA);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				data.push_back(255);
				padding_needed -= 4;
			}
		}
		// Luma
		else if (SIXB(vg) && FOURB(vg_r) && FOURB(vg_b)) {
			if (padding_needed < 2) {
				data.push_back(QOI_OP_LUMA | (vg + 32));
				data.push_back((vg_r + 8) << 4 | (vg_b + 8));
			} else if (padding_needed == 2) {
				data.push_back(QOI_OP_RGB);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				padding_needed -= 2;
			} else {
				data.push_back(QOI_OP_RGBA);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				data.push_back(255);
				padding_needed -= 3;
			}
		}
		// RGB
		else {
			if (padding_needed == 0) {
				data.push_back(QOI_OP_RGB);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
			} else {
				data.push_back(QOI_OP_RGBA);
				data.push_back(px.rgba.r);
				data.push_back(px.rgba.g);
				data.push_back(px.rgba.b);
				data.push_back(255);
				padding_needed--;
			}
		}

		index[index_pos] = px;
		px_prev = px;

		// Segment logic
		int runover = B & segment;
		if (runover == 0 && attempted) {
			attempted = false;
		}

		if (err_cnt == 0 && (last_b & segment) > runover && runover != 0) {
			if (padding_needed == 0 && !attempted) {
				p = last_seg_p;
				padding_needed = segment - (last_b & segment) + 1;
				data.resize(data.size() - runover - segment - 1);
				last_b = B;
				attempted = true;
			} else {
				padding_needed = 0;
				err_cnt++;
				attempted = false;
			}
		}
	}

	// std::cout << "Errors: " << err_cnt << "\tSegments: " << (B / segment) + 1
	// 		  << std::endl;


	// Write footer
	for (auto b : qoi_padding) {
		data.push_back(b);
	}

	return data;
}




template <int CHANNELS>
__global__ void decode_segment(
    uint8_t* pixels,
    uint8_t* bytes,
    size_t total_size,
	size_t total_threads,
    uint8_t* thread_pixel_offsets
) {
    int entry_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (entry_idx >= total_threads) return;

    size_t start_byte = entry_idx * segment;
    size_t end_byte   = start_byte + segment;

    if (start_byte >= total_size) return;
    if (end_byte > total_size) end_byte = total_size;
    
    size_t num_bytes = end_byte - start_byte;

	// change back to just bytes readdreessing
    const uint8_t* segment_bytes = bytes + start_byte;

	
	const size_t max_segment_bytes = (segment + 1) * 64;



    qoi_rgba_t index[64];
    qoi_rgba_t px;
    int run = 0;
    size_t p = 0;
    

	Vector<uint8_t, max_segment_bytes> segment_pxs;



    memset(index, 0, sizeof(index));
    px.v = 0xFF000000u;

    while(p < num_bytes) {

		// if (entry_idx == 1) {
		// 	printf("px: %i\n", px_cnt);
		// }
        if (run > 0) {
            run--;
        } else {
            int b1 = segment_bytes[p++];

            if (b1 == QOI_OP_RGB) {
                if (p + 2 >= num_bytes) break;
                px.rgba.r = segment_bytes[p++];
                px.rgba.g = segment_bytes[p++];
                px.rgba.b = segment_bytes[p++];
            } else if (b1 == QOI_OP_RGBA) {
                if (p + 3 >= num_bytes) break;
                px.rgba.r = segment_bytes[p++];
                px.rgba.g = segment_bytes[p++];
                px.rgba.b = segment_bytes[p++];
                px.rgba.a = segment_bytes[p++];
            } else if ((b1 & QOI_MASK_2) == QOI_OP_INDEX) {
                px = index[b1];
            } else if ((b1 & QOI_MASK_2) == QOI_OP_DIFF) {
                px.rgba.r += ((b1 >> 4) & 0x03) - 2;
                px.rgba.g += ((b1 >> 2) & 0x03) - 2;
                px.rgba.b += (b1 & 0x03) - 2;
            } else if ((b1 & QOI_MASK_2) == QOI_OP_LUMA) {
                if (p >= num_bytes) break;
                int b2 = segment_bytes[p++];
                int vg = (b1 & 0x3f) - 32;
                px.rgba.r += vg - 8 + ((b2 >> 4) & 0x0f);
                px.rgba.g += vg;
                px.rgba.b += vg - 8 + (b2 & 0x0f);
            } else if ((b1 & QOI_MASK_2) == QOI_OP_RUN) {
                run = (b1 & 0x3f);
            }

            index[QOI_COLOR_HASH(px) & 63] = px;
        }

        segment_pxs.push_back(px.rgba.r);
        segment_pxs.push_back(px.rgba.g);
        segment_pxs.push_back(px.rgba.b);
        if constexpr (CHANNELS == 4) {
            segment_pxs.push_back(px.rgba.a);
        }
    }


	//Scan Hills  and Steele reduction thingy
	int pout = 0, pin = 1;
	thread_pixel_offsets[entry_idx] = segment_pxs.size();
	__syncthreads();
	for (int offset = 1; offset < total_threads; offset *= 2) {
		pout = 1 - pout;
		pin = 1 - pout;
		if (entry_idx - offset >= 0) {
			thread_pixel_offsets[pout * total_threads + entry_idx] = thread_pixel_offsets[pin * total_threads + entry_idx] + thread_pixel_offsets[pin * total_threads + entry_idx - offset];
		} else {
			thread_pixel_offsets[pout * total_threads + entry_idx] = thread_pixel_offsets[pin * total_threads + entry_idx];
		}
		__syncthreads(); 
	}
	thread_pixel_offsets[entry_idx] = thread_pixel_offsets[pout * total_threads + entry_idx]; 

	__syncthreads(); 

	int px_offset = thread_pixel_offsets[entry_idx];
	for(unsigned int i = 0; i < segment_pxs.size(); i++) {
		pixels[px_offset + i] = segment_pxs[i];
	}
}





std::vector<uint8_t> GPUQOI::decode(
    const std::vector<uint8_t>& encoded_data,
    QOIDecoderSpec& spec
) {
	size_t encoded_data_size = encoded_data.size();

    // Read in header
    if (encoded_data_size < sizeof(QOIHeader) + sizeof(qoi_padding)) {
        return {};
    }

    QOIHeader header;
    memcpy(header.b, encoded_data.data(), sizeof(QOIHeader));
    // Big Endian :(
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



	uint8_t* pixels = nullptr;
	size_t px_bytes = px_len * sizeof(uint8_t);
	cudaMallocManaged(
		&pixels,
		px_bytes
	);

	int device = 0;
	cudaGetDevice(&device);

	cudaMemLocation loc{};
	loc.type = cudaMemLocationTypeDevice;
	loc.id   = device;

	cudaMemPrefetchAsync(
		pixels,
		px_bytes,
		loc,
		0
	);
	cudaDeviceSynchronize();


    const int threadsPerBlock = 32;
	const int num_entry_points = (max_size + segment - 1) / segment;
    const int blocksPerGrid = (num_entry_points + threadsPerBlock - 1) / threadsPerBlock;

    uint8_t* device_encoded_data = nullptr;
	size_t encoded_data_bytes = encoded_data_size * sizeof(uint8_t);

    cudaMalloc(
		(void**)&device_encoded_data,
		encoded_data_bytes
	);
    cudaMemcpy(
        device_encoded_data,
        encoded_data.data(),
        encoded_data_bytes,
        cudaMemcpyHostToDevice
    );

	size_t total_threads = (blocksPerGrid * threadsPerBlock);

    uint8_t* device_thread_pixel_offsets = nullptr;
	size_t thread_pixel_offsets_bytes = total_threads * sizeof(uint8_t);

	cudaMalloc(
		(void**)&device_thread_pixel_offsets,
		thread_pixel_offsets_bytes
	);


	if (channels == 3) {
		decode_segment<3><<<blocksPerGrid, threadsPerBlock>>>(
			pixels,
			device_encoded_data,
			encoded_data_size,
			total_threads,
			device_thread_pixel_offsets

		);
	} else {
		decode_segment<4><<<blocksPerGrid, threadsPerBlock>>>(
			pixels,
			device_encoded_data,
			encoded_data_size,
			total_threads,
			device_thread_pixel_offsets
		);
    }

	cudaMemPrefetchAsync(
		pixels, 
		px_bytes, 
		loc,
		0
	);
	cudaDeviceSynchronize(); 

    cudaFree(device_encoded_data);

	std::vector<uint8_t> out_pixels(pixels, pixels + px_len);
	cudaFree(pixels);
	return out_pixels;
}

