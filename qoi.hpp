#pragma once

#include <vector>
#include <cstdint>

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
#define QOI_PIXELS_MAX ((unsigned int)400000000)
#define MIN_CHECKPOINT_INTERVAL (1 << 18)  // 256KB
#define PIXEL_TO_ENCODE_RATIO 1.8	 // avg ratio of encoded size to pixels
#define INVALID_PIXEL 0xFF'FF'FF'00	 // 0 alpha pure white

struct QOIDecoderSpec {
	uint32_t width;
	uint32_t height;
	uint8_t channels;
	uint8_t colorspace;
};

class IDecoder {
   public:
	virtual ~IDecoder() = default;
	virtual std::vector<uint8_t> decode(
		const std::vector<uint8_t>& encoded_data,
		QOIDecoderSpec& spec) = 0;
};

struct QOIEncoderSpec {
	uint32_t width;
	uint32_t height;
	uint8_t channels;
	uint8_t colorspace;
};

class IEncoder {
   public:
	virtual ~IEncoder() = default;
	virtual std::vector<uint8_t> encode(const std::vector<uint8_t>& data,
										QOIEncoderSpec& spec) = 0;
};

union QOIHeader {
	struct {
		uint32_t magic;
		uint32_t width;
		uint32_t height;
		uint8_t channels;
		uint8_t colorspace;
	} __attribute__((packed)) fields;
	uint8_t b[14];
};