#include "libpng.hpp"
#include <cstring>
#include <png.h>
#include <vector>
#include <stdexcept>

struct PngReadState {
    const uint8_t* data;
    size_t offset;
};

static void read_data_from_vector(png_structp png_ptr, png_bytep out_bytes, png_size_t byte_count_to_read) {
    PngReadState* read_state = (PngReadState*)png_get_io_ptr(png_ptr);
    memcpy(out_bytes, read_state->data + read_state->offset, byte_count_to_read);
    read_state->offset += byte_count_to_read;
}

std::vector<uint8_t> Libpng::decode(const std::vector<uint8_t>& encoded_data, QOIDecoderSpec& spec) {
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        throw std::runtime_error("Could not create png read struct");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        throw std::runtime_error("Could not create png info struct");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        throw std::runtime_error("Error during png decoding");
    }

    PngReadState read_state = {encoded_data.data(), 0};
    png_set_read_fn(png_ptr, &read_state, read_data_from_vector);

    png_read_info(png_ptr, info_ptr);

    // Apply transformations to ensure 8-bit RGBA output
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY && png_get_bit_depth(png_ptr, info_ptr) < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
    }
    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB ||
        png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY ||
        png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    }
    if (png_get_bit_depth(png_ptr, info_ptr) == 16) {
        png_set_strip_16(png_ptr);
    }

    png_read_update_info(png_ptr, info_ptr);

    spec.width = png_get_image_width(png_ptr, info_ptr);
    spec.height = png_get_image_height(png_ptr, info_ptr);
    spec.channels = 4; // Force to 4 channels (RGBA)
    spec.colorspace = 0; // libpng doesn't have a concept of colorspace in the same way as QOI

    png_uint_32 row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    std::vector<uint8_t> image_data(row_bytes * spec.height);
    std::vector<png_bytep> row_pointers(spec.height);

    for (unsigned int i = 0; i < spec.height; ++i) {
        row_pointers[i] = image_data.data() + i * row_bytes;
    }

    png_read_image(png_ptr, row_pointers.data());

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    return image_data;
}

struct PngWriteState {
    std::vector<uint8_t>* data;
};

static void write_data_to_vector(png_structp png_ptr, png_bytep data, png_size_t length) {
    PngWriteState* write_state = (PngWriteState*)png_get_io_ptr(png_ptr);
    write_state->data->insert(write_state->data->end(), data, data + length);
}

std::vector<uint8_t> Libpng::encode(const std::vector<uint8_t>& data, QOIEncoderSpec& spec) {
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        throw std::runtime_error("Could not create png write struct");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        throw std::runtime_error("Could not create png info struct");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        throw std::runtime_error("Error during png encoding");
    }

    std::vector<uint8_t> encoded_data;
    PngWriteState write_state = {&encoded_data};
    png_set_write_fn(png_ptr, &write_state, write_data_to_vector, NULL);

    // Always write as RGBA
    png_set_IHDR(png_ptr, info_ptr, spec.width, spec.height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<png_bytep> row_pointers(spec.height);
    for (unsigned int i = 0; i < spec.height; ++i) {
        row_pointers[i] = (png_bytep)data.data() + i * spec.width * 4; // Always assume 4 channels for input data
    }

    png_set_rows(png_ptr, info_ptr, row_pointers.data());
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);

    return encoded_data;
}