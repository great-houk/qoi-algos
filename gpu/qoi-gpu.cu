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
static const size_t SEGMENT = 1 << 9;  // Must be a power of 2
static const size_t SEGMENT_MASK = SEGMENT - 1;
static const uint8_t SEGMENT_FOOTER[] = {'S', 'E', 'G', 'S'};
static const int threadsPerBlock = 128;

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
	std::vector<uint32_t> segment_offsets;
	segment_offsets.reserve((data.size() + SEGMENT - 1) / SEGMENT);

	int err_cnt = 0;
	bool attempted = false;

	int p = 0;

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
			last_seg_p = p;
			if (segment_offsets.empty() ||
				segment_offsets[segment_offsets.size() - 1] != p) {
				segment_offsets.push_back(p);
			}
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
	for (auto o : segment_offsets) {
		uint8_t* ob = (uint8_t*)&o;
		for (int i = 0; i < 4; i++) {
			data.push_back(ob[i]);
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

__global__ void decode_into_segment(uint8_t* bytes,
									size_t data_size,
									size_t* pixel_amounts,
									qoi_rgba_t* seg_pixels_g,
									size_t num_segs) {
	size_t entry_idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = num_segs;
	if (entry_idx >= num_segs || (entry_idx * SEGMENT) >= data_size)
		return;

	qoi_rgba_t* seg_pixels = &seg_pixels_g[entry_idx];
	uint8_t* data = &bytes[entry_idx * SEGMENT];
	size_t n_bytes = (entry_idx + 1) * SEGMENT < data_size
						 ? SEGMENT
						 : data_size - (entry_idx * SEGMENT);

	size_t b = 0, p = 0;
	uint8_t run = 0;
	uint8_t lightness = entry_idx % 128;
	qoi_rgba_t px = {.v = 0};
	px.rgba.a = lightness + 128;
	uint8_t index[64 * 3];
	memset(index, 0, sizeof(index));

	while (b < n_bytes) {
		if (run == 0) {
			uint8_t cmd = data[b++];

			if (cmd == QOI_OP_RGB || cmd == QOI_OP_RGBA) {
				px.rgba.r = data[b++];
				px.rgba.g = data[b++];
				px.rgba.b = data[b++];
				b += cmd == QOI_OP_RGBA;
			} else if ((cmd & QOI_MASK_2) == QOI_OP_INDEX) {
				px.rgba.r = index[cmd * 3 + 0];
				px.rgba.g = index[cmd * 3 + 1];
				px.rgba.b = index[cmd * 3 + 2];
			} else if ((cmd & QOI_MASK_2) == QOI_OP_DIFF) {
				px.rgba.r += ((cmd >> 4) & 0x03) - 2;
				px.rgba.g += ((cmd >> 2) & 0x03) - 2;
				px.rgba.b += (cmd & 0x03) - 2;
			} else if ((cmd & QOI_MASK_2) == QOI_OP_LUMA) {
				uint8_t b2 = data[b++];
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
			index[hash * 3 + 0] = px.rgba.r;
			index[hash * 3 + 1] = px.rgba.g;
			index[hash * 3 + 2] = px.rgba.b;
		} else {
			run--;
		}

		seg_pixels[stride * p++] = px;
	}

	for (; run > 0; run--) {
		seg_pixels[stride * p++] = px;
	}

	pixel_amounts[entry_idx] = p;
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

	size_t offsets_idx = encoded_data.size() - 4 - 4 - num_segs * 4;
	for (int i = 0; i < num_segs; i++) {
		uint32_t offset;
		memcpy(&offset, &encoded_data[offsets_idx + i * 4], 4);
		segment_offsets.push_back(offset);
	}

	std::vector<uint8_t> pixels;
	pixels.resize(spec.width * spec.height * spec.channels);

#pragma omp parallel for schedule(dynamic)
	for (int s = 0; s < num_segs; s++) {
		size_t data_len = offsets_idx - sizeof(qoi_padding) - sizeof(QOIHeader);
		size_t n_bytes =
			((s + 1) * SEGMENT) < data_len ? SEGMENT : data_len - (s * SEGMENT);
		const uint8_t* data = &encoded_data[sizeof(QOIHeader) + SEGMENT * s];
		uint8_t* pxs = &pixels[segment_offsets[s]];

		size_t b = 0, p = 0;
		uint8_t run = 0;
		qoi_rgba_t px = {.v = 0xFF000000};
		uint8_t index[64 * 3];
		memset(index, 0, sizeof(index));

		while (b < n_bytes) {
			if (run == 0) {
				uint8_t cmd = data[b++];

				if (cmd == QOI_OP_RGB || cmd == QOI_OP_RGBA) {
					px.rgba.r = data[b++];
					px.rgba.g = data[b++];
					px.rgba.b = data[b++];
					b += cmd == QOI_OP_RGBA;
				} else if ((cmd & QOI_MASK_2) == QOI_OP_INDEX) {
					px.rgba.r = index[cmd * 3 + 0];
					px.rgba.g = index[cmd * 3 + 1];
					px.rgba.b = index[cmd * 3 + 2];
				} else if ((cmd & QOI_MASK_2) == QOI_OP_DIFF) {
					px.rgba.r += ((cmd >> 4) & 0x03) - 2;
					px.rgba.g += ((cmd >> 2) & 0x03) - 2;
					px.rgba.b += (cmd & 0x03) - 2;
				} else if ((cmd & QOI_MASK_2) == QOI_OP_LUMA) {
					uint8_t b2 = data[b++];
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
				index[hash * 3 + 0] = px.rgba.r;
				index[hash * 3 + 1] = px.rgba.g;
				index[hash * 3 + 2] = px.rgba.b;
			} else {
				run--;
			}

			pxs[p++] = px.rgba.r;
			pxs[p++] = px.rgba.g;
			pxs[p++] = px.rgba.b;
			if (spec.channels == 4) {
				pxs[p++] = px.rgba.a;
			}
		}

		for (; run > 0; run--) {
			pxs[p++] = px.rgba.r;
			pxs[p++] = px.rgba.g;
			pxs[p++] = px.rgba.b;
			if (spec.channels == 4) {
				pxs[p++] = px.rgba.a;
			}
		}
	}

	return pixels;

	// const uint8_t* data_start = encoded_data.data() + sizeof(QOIHeader);
	// // Assume there's no extra data, makes this incompatible with our other
	// // encoding methods
	// const size_t data_size =
	// 	encoded_data.size() - sizeof(QOIHeader) - sizeof(qoi_padding);

	// const int numSegments = (data_size + SEGMENT - 1) / SEGMENT;
	// const int numBlocks = (numSegments + threadsPerBlock - 1) /
	// threadsPerBlock;

	// std::vector<size_t> pixel_amounts;
	// pixel_amounts.resize(numSegments);

	// std::vector<qoi_rgba_t> seg_pixels;
	// seg_pixels.resize(SEGMENT_PIXEL_AMOUNT * numSegments, {0});

	// 	uint8_t* gpu_encoded_data;
	// 	CHECK(cudaMalloc(&gpu_encoded_data, data_size));
	// 	CHECK(cudaMemcpy(gpu_encoded_data, data_start, data_size,
	// 					 cudaMemcpyHostToDevice));

	// 	size_t* gpu_pixel_amounts;
	// 	CHECK(cudaMalloc(&gpu_pixel_amounts, numSegments * sizeof(size_t)));

	// 	qoi_rgba_t* gpu_seg_pixels;
	// 	CHECK(cudaMalloc(&gpu_seg_pixels,
	// 					 SEGMENT_PIXEL_AMOUNT * numSegments *
	// sizeof(qoi_rgba_t)));

	// 	decode_into_segment<<<numBlocks, threadsPerBlock>>>(
	// 		gpu_encoded_data, data_size, gpu_pixel_amounts, gpu_seg_pixels,
	// 		numSegments);
	// 	CHECK(cudaDeviceSynchronize());

	// 	CHECK(cudaMemcpy(seg_pixels.data(), gpu_seg_pixels,
	// 					 seg_pixels.size() * sizeof(qoi_rgba_t),
	// 					 cudaMemcpyDeviceToHost));

	// 	CHECK(cudaMemcpy(pixel_amounts.data(), gpu_pixel_amounts,
	// 					 pixel_amounts.size() * sizeof(size_t),
	// 					 cudaMemcpyDeviceToHost));

	// 	auto pixel_offsets = pixel_amounts;
	// 	pixel_offsets[0] = 0;
	// 	for (int i = 1; i < pixel_offsets.size(); i++) {
	// 		pixel_offsets[i] = pixel_offsets[i - 1] + pixel_amounts[i - 1];
	// 	}

	// 	cudaFree(gpu_encoded_data);
	// 	cudaFree(gpu_pixel_amounts);
	// 	cudaFree(gpu_seg_pixels);

	// 	std::vector<uint8_t> pixels;
	// 	pixels.resize(px_len);

	// #pragma omp parallel for schedule(dynamic)
	// 	for (int t = 0; t < numSegments; t++) {
	// 		size_t p = pixel_offsets[t] * CHANNELS;
	// 		for (int i = 0; i < pixel_amounts[t]; i++) {
	// 			qoi_rgba_t px = seg_pixels[t + numSegments * i];

	// 			pixels[p++] = px.rgba.r;
	// 			pixels[p++] = px.rgba.g;
	// 			pixels[p++] = px.rgba.b;
	// 			if (CHANNELS == 4) {
	// 				pixels[p++] = 255;
	// 			}
	// 		}
	// 	}

	// 	return pixels;
}
