#pragma once

#include <vector>
#include <cstdint>

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