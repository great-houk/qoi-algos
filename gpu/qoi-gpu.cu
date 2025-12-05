#include <stdbool.h>
#include <vector>
#include <stdio.h>
#include "qoi-gpu.hpp"

typedef union {
	struct {
		uint8_t r, g, b, a;
	} rgba;
	uint32_t v;
} qoi_rgba_t;

static const uint8_t qoi_padding[8] = {0, 0, 0, 0, 0, 0, 0, 1};
static const size_t SEGMENT = 1 << 10;	// Must be a power of 2
static const size_t SEGMENT_MASK = SEGMENT - 1;
static const uint8_t SEGMENT_FOOTER[] = {'S', 'E', 'G', 'S'};
static const int threadsPerBlock = 128;
static const int sharedDataCacheSize = 1 << 7;

#define CHECK(stmt)                                                 \
	{                                                               \
		cudaError_t result = stmt;                                  \
		if (result != cudaSuccess) {                                \
			printf("Cuda error: %s\n", cudaGetErrorString(result)); \
		}                                                           \
	}

std::vector<uint8_t> GPUQOI::encode(const std::vector<uint8_t>& pixels,
									QOIEncoderSpec& spec) {
	// Allocate
	std::vector<uint8_t> data;
	// Each pixel takes ~1.8 bytes on avg to encode
	// data.reserve(sizeof(QOIHeader) + spec.width * spec.height * 2 +
	// 			 sizeof(qoi_padding) + (spec.width * spec.height / SEGMENT) +
	// 			 sizeof(SEGMENT_FOOTER));

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
	auto B = [&] { return data.size() - sizeof(QOIHeader); };
	std::vector<uint16_t> segment_offsets;
	segment_offsets.reserve((data.size() + SEGMENT - 1) / SEGMENT);

	int err_cnt = 0;
	bool attempted = false;

	uint32_t p = 0;

	qoi_rgba_t index[64];
	qoi_rgba_t px_prev;
	qoi_rgba_t px = {.v = 0xFF'00'00'00};  // Full alpha black

	auto pad = [&](int savings) {
		if (padding_needed == savings) {
			data.push_back(QOI_OP_RGB);
			data.push_back(px.rgba.r);
			data.push_back(px.rgba.g);
			data.push_back(px.rgba.b);
			padding_needed -= savings;
		} else {
			data.push_back(QOI_OP_RGBA);
			data.push_back(px.rgba.r);
			data.push_back(px.rgba.g);
			data.push_back(px.rgba.b);
			data.push_back(255);
			padding_needed -= savings + 1;
		}
	};

	while (p < pixels.size()) {
		// Segment logic
		if ((B() & SEGMENT_MASK) == 0) {
			memset(index, 0, sizeof(index));
			px_prev.v = INVALID_PIXEL;

			if (segment_offsets.empty()) {
				segment_offsets.push_back(0);
			} else {
				if (last_seg_p != p) {
					segment_offsets.push_back(p - last_seg_p);
				}
			}

			last_seg_p = p;
		}
		int last_b = B();

		// Load Pixel
		memcpy(&px.v, &pixels[p], 3);
		p += spec.channels;

		// Calculate differences
		auto TWOB = [](auto x) { return x >= -2 && x <= 1; };
		auto FOURB = [](auto x) { return x >= -8 && x <= 7; };
		auto SIXB = [](auto x) { return x >= -32 && x <= 31; };
		int8_t vr = px.rgba.r - px_prev.rgba.r;
		int8_t vg = px.rgba.g - px_prev.rgba.g;
		int8_t vb = px.rgba.b - px_prev.rgba.b;

		int8_t vg_r = vr - vg;
		int8_t vg_b = vb - vg;

		// Calculate hash
		uint8_t index_pos = QOI_COLOR_HASH(px) & (64 - 1);

		// Invalid previous
		if (px_prev.v == INVALID_PIXEL) {
			pad(0);
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
			} else {
				pad(3);
			}
		}
		// Hash
		else if (index[index_pos].v == px.v) {
			if (padding_needed < 3) {
				data.push_back(QOI_OP_INDEX | index_pos);
			} else {
				pad(3);
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
			} else {
				pad(3);
			}
		}
		// Luma
		else if (SIXB(vg) && FOURB(vg_r) && FOURB(vg_b)) {
			if (padding_needed < 2) {
				data.push_back(QOI_OP_LUMA | (vg + 32));
				data.push_back((vg_r + 8) << 4 | (vg_b + 8));
			} else {
				pad(2);
			}
		}
		// RGB
		else {
			pad(0);
		}

		index[index_pos] = px;
		px_prev = px;

		// Segment logic
		int runover = B() & SEGMENT_MASK;
		if (runover == 0 && attempted) {
			attempted = false;
		}

		if (err_cnt == 0 && (last_b & SEGMENT_MASK) > runover && runover != 0) {
			if (padding_needed == 0 && !attempted) {
				p = last_seg_p;
				padding_needed = SEGMENT - (last_b & SEGMENT_MASK);
				data.resize(data.size() - runover - SEGMENT);
				attempted = true;
			} else {
				padding_needed = 0;
				err_cnt++;
				attempted = false;
			}
		}
	}

	if (err_cnt != 0) {
		printf("Errors: %i\tSegments: %lu\n", err_cnt,
			   (B() / SEGMENT_MASK) + 1);
	}

	// Write footer
	for (auto b : qoi_padding) {
		data.push_back(b);
	}

	// Write segment offsets
	for (int i = 1; i < segment_offsets.size(); i++) {
		uint8_t* bytes = (uint8_t*)&segment_offsets[i];
		for (int j = 0; j < 2; j++) {
			data.push_back(bytes[j]);
		}
	}

	// Write number segments
	uint32_t len = segment_offsets.size();
	uint8_t* bytes = (uint8_t*)&len;
	for (int i = 0; i < 4; i++) {
		data.push_back(bytes[i]);
	}

	// Write footer
	for (auto b : SEGMENT_FOOTER) {
		data.push_back(b);
	}

	return data;
}

template <int CHANNELS>
__global__ void decode_into_segment(uint8_t* data_g,
									size_t data_size,
									uint32_t* segment_offsets,
									size_t num_segs,
									uint8_t* pixels_g) {
	size_t entry_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (entry_idx >= num_segs)
		return;

	uint8_t* pixels = &pixels_g[segment_offsets[entry_idx]];
	uint8_t* data = &data_g[entry_idx * SEGMENT];
	size_t n_bytes = (entry_idx + 1) * SEGMENT < data_size
						 ? SEGMENT
						 : data_size - (entry_idx * SEGMENT);

	size_t b = 0, b_s = 0, p = 0;
	uint8_t run = 0;
	qoi_rgba_t px = {.v = 0};
	px.rgba.a = 0xFF;

	qoi_rgba_t index[64];
	memset(index, 0, sizeof(index));

	__shared__ uint8_t data_s[threadsPerBlock * sharedDataCacheSize];
	auto cache_data = [&] {
		auto num = b + sharedDataCacheSize >= n_bytes ? n_bytes - b
													  : sharedDataCacheSize;
		memcpy(&data_s[threadIdx.x * sharedDataCacheSize], &data[b], num);
	};
	cache_data();

	auto get_data = [&] {
		b++, b_s++;
		return data_s[(b_s - 1) + threadIdx.x * sharedDataCacheSize];
	};

	while (b < n_bytes) {
		if (b_s >= sharedDataCacheSize - 4) {
			b_s = 0;
			cache_data();
		}

		if (run == 0) {
			uint8_t cmd = get_data();

			if (cmd == QOI_OP_RGB || cmd == QOI_OP_RGBA) {
				px.rgba.r = get_data();
				px.rgba.g = get_data();
				px.rgba.b = get_data();
				b += cmd == QOI_OP_RGBA;
				b_s += cmd == QOI_OP_RGBA;
			} else if ((cmd & QOI_MASK_2) == QOI_OP_INDEX) {
				px = index[cmd];
			} else if ((cmd & QOI_MASK_2) == QOI_OP_DIFF) {
				px.rgba.r += ((cmd >> 4) & 0x03) - 2;
				px.rgba.g += ((cmd >> 2) & 0x03) - 2;
				px.rgba.b += (cmd & 0x03) - 2;
			} else if ((cmd & QOI_MASK_2) == QOI_OP_LUMA) {
				uint8_t b2 = get_data();
				uint8_t vg = (cmd & 0x3f) - 32;
				px.rgba.r += vg - 8 + ((b2 >> 4) & 0x0f);
				px.rgba.g += vg;
				px.rgba.b += vg - 8 + (b2 & 0x0f);
			} else if ((cmd & QOI_MASK_2) == QOI_OP_RUN) {
				run = cmd & 0x3f;
			}

			uint8_t hash =
				(px.rgba.r * 3 + px.rgba.g * 5 + px.rgba.b * 7 + 245) &
				(64 - 1);
			index[hash] = px;
		} else {
			run--;
		}

		pixels[p++] = px.rgba.r;
		pixels[p++] = px.rgba.g;
		pixels[p++] = px.rgba.b;
		if constexpr (CHANNELS == 4) {
			pixels[p++] = px.rgba.a;
		}
	}

	for (; run > 0; run--) {
		pixels[p++] = px.rgba.r;
		pixels[p++] = px.rgba.g;
		pixels[p++] = px.rgba.b;
		if constexpr (CHANNELS == 4) {
			pixels[p++] = px.rgba.a;
		}
	}
}

std::vector<uint8_t> GPUQOI::decode(const std::vector<uint8_t>& encoded_data,
									QOIDecoderSpec& spec) {
	// Read in header
	if (encoded_data.size() < sizeof(QOIHeader) + sizeof(qoi_padding)) {
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

	uint8_t CHANNELS = spec.channels;
	size_t px_len = spec.width * spec.height * CHANNELS;

	// Read in footer and segment offsets
	uint8_t footer[4];
	memcpy(&footer, &encoded_data[encoded_data.size() - 4], 4);
	if (memcmp(&footer, &SEGMENT_FOOTER, 4) != 0) {
		return {};
	}
	uint32_t num_segs;
	memcpy(&num_segs, &encoded_data[encoded_data.size() - 4 - 4], 4);

	std::vector<uint32_t> segment_offsets;
	segment_offsets.reserve(num_segs);
	segment_offsets.push_back(0);

	size_t offsets_idx = encoded_data.size() - 4 - 4 - (num_segs - 1) * 2;
	for (int i = 0; i < num_segs - 1; i++) {
		uint16_t offset;
		memcpy(&offset, &encoded_data[offsets_idx + i * 2], 2);

		auto tail = segment_offsets[segment_offsets.size() - 1];
		segment_offsets.push_back(offset + tail);
	}

	size_t data_len = offsets_idx - sizeof(qoi_padding) - sizeof(QOIHeader);
	uint8_t* gpu_data = 0;
	CHECK(cudaMalloc(&gpu_data, data_len));
	CHECK(cudaMemcpy(gpu_data, &encoded_data[sizeof(QOIHeader)], data_len,
					 cudaMemcpyHostToDevice));

	std::vector<uint8_t> pixels;
	uint8_t* gpu_pixels = 0;
	CHECK(cudaMalloc(&gpu_pixels, px_len));

	uint32_t* gpu_seg_offsets = 0;
	CHECK(cudaMalloc(&gpu_seg_offsets, num_segs * sizeof(uint32_t)));
	CHECK(cudaMemcpy(gpu_seg_offsets, segment_offsets.data(),
					 num_segs * sizeof(uint32_t), cudaMemcpyHostToDevice));

	const int numBlocks = (num_segs + threadsPerBlock - 1) / threadsPerBlock;

	if (spec.channels == 3) {
		decode_into_segment<3><<<numBlocks, threadsPerBlock>>>(
			gpu_data, data_len, gpu_seg_offsets, num_segs, gpu_pixels);
	} else {
		decode_into_segment<4><<<numBlocks, threadsPerBlock>>>(
			gpu_data, data_len, gpu_seg_offsets, num_segs, gpu_pixels);
	}

	pixels.resize(px_len);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(
		cudaMemcpy(pixels.data(), gpu_pixels, px_len, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(gpu_data));
	CHECK(cudaFree(gpu_pixels));
	CHECK(cudaFree(gpu_seg_offsets));

	return pixels;
}
