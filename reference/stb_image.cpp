#include "stb_image.hpp"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

std::vector<uint8_t> StbImage::decode(const std::vector<uint8_t>& encoded_data,
									  QOIDecoderSpec& spec) {
	int width, height, channels;
	unsigned char* img =
		stbi_load_from_memory(encoded_data.data(), encoded_data.size(), &width,
							  &height, &channels, 0);
	if (!img) {
		throw std::runtime_error("Failed to decode image with stb_image");
	}

	spec.width = width;
	spec.height = height;
	spec.channels = channels;
	spec.colorspace = 0;  // stb_image doesn't have a concept of colorspace in
						  // the same way as QOI

	std::vector<uint8_t> decoded_data(img, img + width * height * channels);
	stbi_image_free(img);
	return decoded_data;
}

static void write_func(void* context, void* data, int size) {
	std::vector<uint8_t>* buffer = static_cast<std::vector<uint8_t>*>(context);
	buffer->insert(buffer->end(), static_cast<uint8_t*>(data),
				   static_cast<uint8_t*>(data) + size);
}

std::vector<uint8_t> StbImage::encode(const std::vector<uint8_t>& data,
									  QOIEncoderSpec& spec) {
	std::vector<uint8_t> encoded_data;
	int success = stbi_write_png_to_func(
		write_func, &encoded_data, spec.width, spec.height, spec.channels,
		data.data(), spec.width * spec.channels);
	if (!success) {
		throw std::runtime_error("Failed to encode image with stb_image_write");
	}
	return encoded_data;
}
