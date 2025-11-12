#include "qoi-mc.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <omp.h>

#define QOI_OP_INDEX 0x00
#define QOI_OP_DIFF 0x40
#define QOI_OP_LUMA 0x80
#define QOI_OP_RUN 0xc0
#define QOI_OP_RGB 0xfe
#define QOI_OP_RGBA 0xff
#define QOI_MASK_2 0xc0

#define QOI_COLOR_HASH(C) \
	(C.rgba.r * 3 + C.rgba.g * 5 + C.rgba.b * 7 + C.rgba.a * 11)
#define QOI_MAGIC                                                \
	((((unsigned int)'q') << 24) | (((unsigned int)'o') << 16) | \
	 (((unsigned int)'i') << 8) | (((unsigned int)'f')))
#define QOI_HEADER_SIZE 14
#define QOI_PIXELS_MAX ((unsigned int)400000000)
#define CHECKPOINT_INTERVAL (1 << 20)  // 1 MiB
#define INVALID_PIXEL 0xFF'FF'FF'00	   // 0 alpha pure white

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

static void qoi_write_32(uint8_t* bytes, uint32_t v) {
	bytes[0] = (0xff000000 & v) >> 24;
	bytes[1] = (0x00ff0000 & v) >> 16;
	bytes[2] = (0x0000ff00 & v) >> 8;
	bytes[3] = (0x000000ff & v);
}

static unsigned int qoi_read_32(const uint8_t* bytes) {
	unsigned int a = bytes[0];
	unsigned int b = bytes[1];
	unsigned int c = bytes[2];
	unsigned int d = bytes[3];
	return a << 24 | b << 16 | c << 8 | d;
}

static void encode_segment(std::vector<uint8_t>& bytes,
						   const uint8_t* pixels,
						   uint32_t channels,
						   uint32_t num_px) {
	qoi_rgba_t index[64];
	memset(index, INVALID_PIXEL, sizeof(index));
	qoi_rgba_t px_prev = {.v = INVALID_PIXEL};
	qoi_rgba_t px = {.v = 0x00'00'00'FF};  // Full alpha black
	int run = 0;

	bytes.reserve(num_px * 3);

	for (size_t px_pos = 0; px_pos < num_px * channels; px_pos += channels) {
		// loads current pixel RGB
		px.rgba.r = pixels[px_pos + 0];
		px.rgba.g = pixels[px_pos + 1];
		px.rgba.b = pixels[px_pos + 2];

		// loads alpha of current pixel if exists
		if (channels == 4) {
			px.rgba.a = pixels[px_pos + 3];
		}

		// If current pixel is same as previous pixel
		if (px.v == px_prev.v) {
			run++;
			if (run == 62) {
				bytes.push_back(QOI_OP_RUN | (run - 1));
				run = 0;
			}
		} else {
			// if pixel color changed we push to byte and flush run
			if (run > 0) {
				bytes.push_back(QOI_OP_RUN | (run - 1));
				run = 0;
			}

			// Hash current pixel to a color index slot [0..63].
			// basically maping the pixel to a position in the index table
			int index_pos = QOI_COLOR_HASH(px) & (64 - 1);

			// Fast path: if the hash slot holds the exact same pixel, emit
			// INDEX op.
			if (index[index_pos].v != INVALID_PIXEL &&
				index[index_pos].v == px.v) {
				bytes.push_back(QOI_OP_INDEX | index_pos);
			} else {
				// Otherwise, update the index slot with this pixel.
				index[index_pos] = px;

				// If px_prev was invalidated (e.g., at start or after a
				// checkpoint), we must write this pixel absolutely
				// (RGB/RGBA).
				if (px_prev.v == INVALID_PIXEL) {
					if (channels == 4) {
						bytes.push_back(QOI_OP_RGBA);
					} else {
						bytes.push_back(QOI_OP_RGB);
					}

					bytes.push_back(px.rgba.r);
					bytes.push_back(px.rgba.g);
					bytes.push_back(px.rgba.b);
					if (channels == 4) {
						bytes.push_back(px.rgba.a);
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
							bytes.push_back(QOI_OP_DIFF | (vr + 2) << 4 |
											(vg + 2) << 2 | (vb + 2));
						} else if (vg_r > -9 && vg_r < 8 && vg > -33 &&
								   vg < 32 && vg_b > -9 && vg_b < 8) {
							bytes.push_back(QOI_OP_LUMA | (vg + 32));
							bytes.push_back((vg_r + 8) << 4 | (vg_b + 8));
						} else {
							bytes.push_back(QOI_OP_RGB);
							bytes.push_back(px.rgba.r);
							bytes.push_back(px.rgba.g);
							bytes.push_back(px.rgba.b);
						}
					} else {
						bytes.push_back(QOI_OP_RGBA);
						bytes.push_back(px.rgba.r);
						bytes.push_back(px.rgba.g);
						bytes.push_back(px.rgba.b);
						bytes.push_back(px.rgba.a);
					}
				}
			}
		}
		px_prev = px;
	}

	// Flush any remaining run.
	if (run > 0) {
		bytes.push_back(QOI_OP_RUN | (run - 1));
	}
}

std::vector<uint8_t> MultiCPUQOI::encode(const std::vector<uint8_t>& pixels,
										 QOIEncoderSpec& spec) {
	if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}
	int channels = spec.channels;
	int total_px = spec.width * spec.height;

	// Approximate 3 bytes per pixel
	int num_segments = (total_px * 3) / CHECKPOINT_INTERVAL + 1;
	// std::cout << "Encoding with " << num_segments << " segments." <<
	// std::endl;
	// int num_segments = 2;

	// ceiling division == total_px / num_segments rounded up
	int seg_px = (total_px + num_segments - 1) / num_segments;

	std::vector<uint8_t> vecs[num_segments];
	qoi_checkpoint_t checkpoints[num_segments];

	// Process pixels
#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_segments; s++) {
		// Build checkpoint
		const int start_px = s * seg_px * channels;
		const int end_px =
			std::min(total_px * channels, start_px + seg_px * channels);
		// 0 Byte offset since we don't know it yet
		checkpoints[s] = {
			.fields = {0, static_cast<int32_t>(s * seg_px * channels)}};

		// Encode slice
		encode_segment(vecs[s], &pixels[start_px], channels,
					   (end_px - start_px) / channels);
	}

	// Update checkpoints
	uint32_t total_bytes = QOI_HEADER_SIZE;
	for (int s = 0; s < num_segments; s++) {
		checkpoints[s].fields.byte_offset = total_bytes;
		total_bytes += vecs[s].size();
	}

	// Allocate output bytes
	std::vector<uint8_t> bytes;
	bytes.reserve(total_bytes + sizeof(qoi_padding) +
				  num_segments * sizeof(qoi_checkpoint_t) +
				  sizeof(qoi_padding));
	bytes.resize(total_bytes);
	auto bytes_ptr = bytes.data();

	// Fill in header
	qoi_write_32(bytes_ptr, QOI_MAGIC);
	qoi_write_32(bytes_ptr + 4, spec.width);
	qoi_write_32(bytes_ptr + 8, spec.height);
	bytes_ptr[12] = spec.channels;
	bytes_ptr[13] = spec.colorspace;

// Combine segments
#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_segments; s++) {
		// Append segment data
		std::vector<uint8_t>& segment_bytes = vecs[s];
		int start = checkpoints[s].fields.byte_offset;
		std::memcpy(bytes_ptr + start, segment_bytes.data(),
					segment_bytes.size());
	}

	// Append footer, checkpoints, and second footer
	for (auto b : qoi_padding) {
		bytes.push_back(b);
	}
	for (auto c : checkpoints) {
		for (auto b : c.v) {
			bytes.push_back(b);
		}
	}
	for (auto b : qoi_padding) {
		bytes.push_back(b);
	}

	return bytes;
}

std::vector<uint8_t> MultiCPUQOI::decode(
	const std::vector<uint8_t>& encoded_data,
	QOIDecoderSpec& spec) {
	// Read in header
	const unsigned char* bytes = encoded_data.data();
	int p = 0;
	unsigned int header_magic;

	if (encoded_data.size() < QOI_HEADER_SIZE + sizeof(qoi_padding)) {
		return {};
	}

	header_magic = qoi_read_32(&bytes[0]);
	spec.width = qoi_read_32(&bytes[sizeof(uint32_t) * 1]);
	spec.height = qoi_read_32(&bytes[sizeof(uint32_t) * 2]);
	spec.channels = bytes[sizeof(uint32_t) * 3];
	spec.colorspace = bytes[sizeof(uint32_t) * 3 + 1];

	if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 || header_magic != QOI_MAGIC ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}

	int channels = spec.channels;
	int px_len = spec.width * spec.height * channels;

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
	std::vector<checkpoint_span_t> checkpoints;
	int max_size = spec.width * spec.height * (spec.channels + 1) +
				   QOI_HEADER_SIZE + sizeof(qoi_padding);
	int max_num_checkpoints = (max_size / CHECKPOINT_INTERVAL) + 1;
	int last_cp_px_pos = px_len;
	// Read backwards to check for double padding
	bool found_double_padding = false;
	int p_size = (int)sizeof(qoi_padding);
	int c_size = (int)sizeof(qoi_checkpoint_t);
	int cp_p = (int)encoded_data.size() - p_size;
	for (int i = 0; i < max_num_checkpoints; i++) {
		bool is_padding = true;
		for (int j = 0; j < (int)sizeof(qoi_padding); j++) {
			if (encoded_data[cp_p - p_size + j] != qoi_padding[j]) {
				is_padding = false;
				break;
			}
		}
		if (is_padding) {
			found_double_padding = true;
			// std::cout << "Found " << i << " checkpoints in QOI data."
			// 		  << std::endl;
			break;
		}
		checkpoint_span_t cp;
		for (int j = 0; j < c_size; j++) {
			cp.v[j] = encoded_data[cp_p - c_size + j];
		}
		// We're reading backwards, so this is right
		cp.fields.next_px_pos = last_cp_px_pos;
		last_cp_px_pos = cp.fields.px_pos;
		checkpoints.push_back(cp);
		cp_p -= sizeof(qoi_checkpoint_t);
	}
	if (!found_double_padding) {
		// No double padding found, reset to no checkpoints
		checkpoints.clear();
		checkpoints.push_back((checkpoint_span_t){.fields = {p, 0, px_len}});
		std::cout << "WARNING: No checkpoints found in QOI data, decoding from "
					 "beginning only."
				  << std::endl;
	}

	// Initialize stable state
	std::vector<uint8_t> pixels;
	pixels.resize(px_len);
	int chunks_len = encoded_data.size() - (int)sizeof(qoi_padding);

#pragma omp parallel for schedule(static)
	for (auto cp : checkpoints) {
		// std::cout << "Decoding from checkpoint at byte offset "
		// 		  << cp.fields.byte_offset << ", pixel position "
		// 		  << cp.fields.px_pos << std::endl;

		// Set per checkpoint variables
		int p_local = cp.fields.byte_offset;
		int px_pos = cp.fields.px_pos;

		// Initalize per checkpoint state
		qoi_rgba_t index[64];
		qoi_rgba_t px;
		int run = 0;

		memset(index, 0, sizeof(index));
		px.rgba.r = 0;
		px.rgba.g = 0;
		px.rgba.b = 0;
		px.rgba.a = 255;

		for (; px_pos < cp.fields.next_px_pos; px_pos += channels) {
			if (run > 0) {
				run--;
			} else if (p_local < chunks_len) {
				int b1 = bytes[p_local++];

				if (b1 == QOI_OP_RGB) {
					px.rgba.r = bytes[p_local++];
					px.rgba.g = bytes[p_local++];
					px.rgba.b = bytes[p_local++];
				} else if (b1 == QOI_OP_RGBA) {
					px.rgba.r = bytes[p_local++];
					px.rgba.g = bytes[p_local++];
					px.rgba.b = bytes[p_local++];
					px.rgba.a = bytes[p_local++];
				} else if ((b1 & QOI_MASK_2) == QOI_OP_INDEX) {
					px = index[b1];
				} else if ((b1 & QOI_MASK_2) == QOI_OP_DIFF) {
					px.rgba.r += ((b1 >> 4) & 0x03) - 2;
					px.rgba.g += ((b1 >> 2) & 0x03) - 2;
					px.rgba.b += (b1 & 0x03) - 2;
				} else if ((b1 & QOI_MASK_2) == QOI_OP_LUMA) {
					int b2 = bytes[p_local++];
					int vg = (b1 & 0x3f) - 32;
					px.rgba.r += vg - 8 + ((b2 >> 4) & 0x0f);
					px.rgba.g += vg;
					px.rgba.b += vg - 8 + (b2 & 0x0f);
				} else if ((b1 & QOI_MASK_2) == QOI_OP_RUN) {
					run = (b1 & 0x3f);
				}

				index[QOI_COLOR_HASH(px) & (64 - 1)] = px;
			}

			pixels[px_pos + 0] = px.rgba.r;
			pixels[px_pos + 1] = px.rgba.g;
			pixels[px_pos + 2] = px.rgba.b;

			if (channels == 4) {
				pixels[px_pos + 3] = px.rgba.a;
			}
		}
	}

	return pixels;
}
