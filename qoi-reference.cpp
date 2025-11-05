#include "qoi-reference.hpp"
#include <cstdlib>
#include <cstring>

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

typedef union {
	struct {
		unsigned char r, g, b, a;
	} rgba;
	unsigned int v;
} qoi_rgba_t;

static const unsigned char qoi_padding[8] = {0, 0, 0, 0, 0, 0, 0, 1};

static void qoi_write_32(unsigned char* bytes, int* p, unsigned int v) {
	bytes[(*p)++] = (0xff000000 & v) >> 24;
	bytes[(*p)++] = (0x00ff0000 & v) >> 16;
	bytes[(*p)++] = (0x0000ff00 & v) >> 8;
	bytes[(*p)++] = (0x000000ff & v);
}

static unsigned int qoi_read_32(const unsigned char* bytes, int* p) {
	unsigned int a = bytes[(*p)++];
	unsigned int b = bytes[(*p)++];
	unsigned int c = bytes[(*p)++];
	unsigned int d = bytes[(*p)++];
	return a << 24 | b << 16 | c << 8 | d;
}

std::vector<uint8_t> ReferenceQOI::encode(const std::vector<uint8_t>& data,
										  QOIEncoderSpec& spec) {
	int p = 0, run = 0;
	int px_len, px_end, px_pos, channels;
	std::vector<uint8_t> bytes;
	const unsigned char* pixels = data.data();
	qoi_rgba_t index[64];
	qoi_rgba_t px, px_prev;

	if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}

	int max_size = spec.width * spec.height * (spec.channels + 1) +
				   QOI_HEADER_SIZE + sizeof(qoi_padding);
	bytes.resize(max_size);

	qoi_write_32(bytes.data(), &p, QOI_MAGIC);
	qoi_write_32(bytes.data(), &p, spec.width);
	qoi_write_32(bytes.data(), &p, spec.height);
	bytes[p++] = spec.channels;
	bytes[p++] = spec.colorspace;

	memset(index, 0, sizeof(index));

	run = 0;
	px_prev.rgba.r = 0;
	px_prev.rgba.g = 0;
	px_prev.rgba.b = 0;
	px_prev.rgba.a = 255;
	px = px_prev;

	px_len = spec.width * spec.height * spec.channels;
	px_end = px_len - spec.channels;
	channels = spec.channels;

	for (px_pos = 0; px_pos < px_len; px_pos += channels) {
		px.rgba.r = pixels[px_pos + 0];
		px.rgba.g = pixels[px_pos + 1];
		px.rgba.b = pixels[px_pos + 2];

		if (channels == 4) {
			px.rgba.a = pixels[px_pos + 3];
		}

		if (px.v == px_prev.v) {
			run++;
			if (run == 62 || px_pos == px_end) {
				bytes[p++] = QOI_OP_RUN | (run - 1);
				run = 0;
			}
		} else {
			int index_pos;

			if (run > 0) {
				bytes[p++] = QOI_OP_RUN | (run - 1);
				run = 0;
			}

			index_pos = QOI_COLOR_HASH(px) & (64 - 1);

			if (index[index_pos].v == px.v) {
				bytes[p++] = QOI_OP_INDEX | index_pos;
			} else {
				index[index_pos] = px;

				if (px.rgba.a == px_prev.rgba.a) {
					signed char vr = px.rgba.r - px_prev.rgba.r;
					signed char vg = px.rgba.g - px_prev.rgba.g;
					signed char vb = px.rgba.b - px_prev.rgba.b;

					signed char vg_r = vr - vg;
					signed char vg_b = vb - vg;

					if (vr > -3 && vr < 2 && vg > -3 && vg < 2 && vb > -3 &&
						vb < 2) {
						bytes[p++] = QOI_OP_DIFF | (vr + 2) << 4 |
									 (vg + 2) << 2 | (vb + 2);
					} else if (vg_r > -9 && vg_r < 8 && vg > -33 && vg < 32 &&
							   vg_b > -9 && vg_b < 8) {
						bytes[p++] = QOI_OP_LUMA | (vg + 32);
						bytes[p++] = (vg_r + 8) << 4 | (vg_b + 8);
					} else {
						bytes[p++] = QOI_OP_RGB;
						bytes[p++] = px.rgba.r;
						bytes[p++] = px.rgba.g;
						bytes[p++] = px.rgba.b;
					}
				} else {
					bytes[p++] = QOI_OP_RGBA;
					bytes[p++] = px.rgba.r;
					bytes[p++] = px.rgba.g;
					bytes[p++] = px.rgba.b;
					bytes[p++] = px.rgba.a;
				}
			}
		}
		px_prev = px;
	}

	for (int i = 0; i < (int)sizeof(qoi_padding); i++) {
		bytes[p++] = qoi_padding[i];
	}

	bytes.resize(p);
	return bytes;
}

std::vector<uint8_t> ReferenceQOI::decode(
	const std::vector<uint8_t>& encoded_data,
	QOIDecoderSpec& spec) {
	const unsigned char* bytes = encoded_data.data();
	unsigned int header_magic;
	std::vector<uint8_t> pixels;
	qoi_rgba_t index[64];
	qoi_rgba_t px;
	int px_len, chunks_len, px_pos;
	int p = 0, run = 0;

	if (encoded_data.size() < QOI_HEADER_SIZE + sizeof(qoi_padding)) {
		return {};
	}

	header_magic = qoi_read_32(bytes, &p);
	spec.width = qoi_read_32(bytes, &p);
	spec.height = qoi_read_32(bytes, &p);
	spec.channels = bytes[p++];
	spec.colorspace = bytes[p++];

	if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 || header_magic != QOI_MAGIC ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}

	int channels = spec.channels;
	px_len = spec.width * spec.height * channels;
	pixels.resize(px_len);

	memset(index, 0, sizeof(index));
	px.rgba.r = 0;
	px.rgba.g = 0;
	px.rgba.b = 0;
	px.rgba.a = 255;

	chunks_len = encoded_data.size() - (int)sizeof(qoi_padding);
	for (px_pos = 0; px_pos < px_len; px_pos += channels) {
		if (run > 0) {
			run--;
		} else if (p < chunks_len) {
			int b1 = bytes[p++];

			if (b1 == QOI_OP_RGB) {
				px.rgba.r = bytes[p++];
				px.rgba.g = bytes[p++];
				px.rgba.b = bytes[p++];
			} else if (b1 == QOI_OP_RGBA) {
				px.rgba.r = bytes[p++];
				px.rgba.g = bytes[p++];
				px.rgba.b = bytes[p++];
				px.rgba.a = bytes[p++];
			} else if ((b1 & QOI_MASK_2) == QOI_OP_INDEX) {
				px = index[b1];
			} else if ((b1 & QOI_MASK_2) == QOI_OP_DIFF) {
				px.rgba.r += ((b1 >> 4) & 0x03) - 2;
				px.rgba.g += ((b1 >> 2) & 0x03) - 2;
				px.rgba.b += (b1 & 0x03) - 2;
			} else if ((b1 & QOI_MASK_2) == QOI_OP_LUMA) {
				int b2 = bytes[p++];
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

	return pixels;
}
