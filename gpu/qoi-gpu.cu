#include <stdbool.h>
#include <vector>
#include <iostream>

#include "qoi-gpu.hpp"

typedef union {
	struct {
		uint8_t r, g, b, a;
	} rgba;
	uint32_t v;
} qoi_rgba_t;

static const uint8_t qoi_padding[8] = {0, 0, 0, 0, 0, 0, 0, 1};

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
	int segment = 0xFFF;
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
			px_prev = {.v = INVALID_PIXEL};
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

	std::cout << "Errors: " << err_cnt << "\tSegments: " << (B / segment) + 1
			  << std::endl;
	// for (auto c : errs) {
	// 	std::cout << c << std::endl;
	// }

	// Write footer
	for (auto b : qoi_padding) {
		data.push_back(b);
	}

	return data;
}

std::vector<uint8_t> GPUQOI::decode(const std::vector<uint8_t>& encoded_data,
									QOIDecoderSpec& spec) {
	return std::vector<uint8_t>();
}