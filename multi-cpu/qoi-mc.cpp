#include "qoi-mc.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <omp.h>

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

static void encode_segment(std::vector<uint8_t>& bytes,
						   const uint8_t* pixels,
						   uint32_t channels,
						   uint32_t num_px) {
	qoi_rgba_t index[64];
	memset(index, INVALID_PIXEL, sizeof(index));
	qoi_rgba_t px_prev = {.v = INVALID_PIXEL};
	qoi_rgba_t px = {.v = 0x00'00'00'FF};  // Full alpha black
	int run = 0;

	// Assume worst case, so we don't need to resize
	bytes.reserve(num_px * (channels + 1));

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

	int num_segments =
		(total_px * PIXEL_TO_ENCODE_RATIO) / this->CHECKPOINT_INTERVAL + 1;
	int checks_per_seg = this->CHECKPOINTS_PER_SEGMENT;

	// ceiling division == total_px / num_segments rounded up
	int seg_px = (total_px + num_segments - 1) / num_segments;

	std::vector<uint8_t> vecs[num_segments];
	qoi_checkpoint_t checkpoints[num_segments];

// Process pixels
#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_segments; s += checks_per_seg) {
		for (int j = 0; j < checks_per_seg && s + j < num_segments; j++) {
			// Build checkpoint
			const int start_px = (s + j) * seg_px * channels;
			const int end_px =
				std::min(total_px * channels, start_px + seg_px * channels);
			// 0 Byte offset since we don't know it yet
			checkpoints[s + j] = {
				.fields = {0,
						   static_cast<int32_t>((s + j) * seg_px * channels)}};

			// Encode slice
			encode_segment(vecs[s + j], &pixels[start_px], channels,
						   (end_px - start_px) / channels);
		}
	}

	// Update checkpoints
	uint32_t data_nbytes = sizeof(QOIHeader);
	for (int s = 0; s < num_segments; s++) {
		checkpoints[s].fields.byte_offset = data_nbytes;
		data_nbytes += vecs[s].size();
	}

	// Allocate output bytes
	uint32_t total_bytes = data_nbytes + sizeof(qoi_padding) +
						   num_segments * sizeof(qoi_checkpoint_t) +
						   sizeof(qoi_padding);
	auto bytes_v = std::vector<uint8_t>(total_bytes);
	uint8_t* bytes = bytes_v.data();

	// Fill in header
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
	std::memcpy(bytes, header.b, sizeof(QOIHeader));

// Combine segments
#pragma omp parallel for schedule(static)
	for (int s = 0; s < num_segments; s++) {
		// Append segment data
		auto& segment_bytes = vecs[s];
		int start = checkpoints[s].fields.byte_offset;
		std::memcpy(bytes + start, segment_bytes.data(), segment_bytes.size());
	}

	// Append footer, checkpoints, and second footer
	std::memcpy(bytes + data_nbytes, qoi_padding, sizeof(qoi_padding));
	int checkpoint_offset = data_nbytes + sizeof(qoi_padding);
	for (auto c : checkpoints) {
		std::memcpy(bytes + checkpoint_offset, c.v, sizeof(c.v));
		checkpoint_offset += sizeof(c.v);
	}
	std::memcpy(bytes + checkpoint_offset, qoi_padding, sizeof(qoi_padding));

	return bytes_v;
}

std::vector<uint8_t> MultiCPUQOI::decode(
	const std::vector<uint8_t>& encoded_data,
	QOIDecoderSpec& spec) {
	// Read in header
	const unsigned char* bytes = encoded_data.data();
	unsigned int header_magic;

	if (encoded_data.size() < sizeof(QOIHeader) + sizeof(qoi_padding)) {
		return {};
	}

	QOIHeader header;
	std::memcpy(header.b, bytes, sizeof(QOIHeader));
	// Big Endian :(
	header_magic = __builtin_bswap32(header.fields.magic);
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
	checkpoints.reserve(max_num_checkpoints);
	// Read backwards to check for double padding
	bool found_double_padding = false;
	int p_size = (int)sizeof(qoi_padding);
	int c_size = (int)sizeof(qoi_checkpoint_t);
	int last_cp_px_pos = px_len;
	const uint8_t* cp_p = encoded_data.data() + encoded_data.size() - p_size;
	for (int i = 0; i < max_num_checkpoints; i++) {
		// Check if we found the second padding
		uint8_t padding[8];
		std::memcpy(padding, cp_p - p_size, p_size);
		if (std::memcmp(padding, qoi_padding, p_size) == 0) {
			found_double_padding = true;
			break;
		}
		// Read in checkpoint
		checkpoint_span_t cp;
		std::memcpy(cp.v, cp_p - c_size, c_size);
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

	// Initialize stable state
	uint8_t* pixels = new uint8_t[px_len];
	int chunks_len = encoded_data.size() - (int)sizeof(qoi_padding);

#pragma omp parallel for schedule(static)
	for (auto cp : checkpoints) {
		// Set per checkpoint variables
		int p_local = cp.fields.byte_offset;
		int px_pos = cp.fields.px_pos;

		// Initalize per checkpoint state
		qoi_rgba_t index[64];
		qoi_rgba_t px;
		int run = 0;

		memset(index, 0, sizeof(index));
		px.v = 0x00'00'00'FF;  // Full alpha black

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

	return std::vector<uint8_t>(pixels, pixels + px_len);
}
