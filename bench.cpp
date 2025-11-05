#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <dirent.h>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "qoi.hpp"
#include "qoi-reference.hpp"
#include "single-cpu/qoi-sc.hpp"

struct Implementation {
    std::string name;
    IEncoder* encoder;
    IDecoder* decoder;
};

int main() {
    std::vector<Implementation> implementations = {
        {"Reference", new ReferenceQOI(), new ReferenceQOI()},
        {"Single CPU", new SingleCPUQOI(), new SingleCPUQOI()}
    };

    std::vector<std::string> image_paths;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir("/home/tyler/qoi-algos/images")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".png") {
                image_paths.push_back("/home/tyler/qoi-algos/images/" + filename);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory" << std::endl;
        return 1;
    }
    std::sort(image_paths.begin(), image_paths.end());

    std::cout << std::left << std::setw(20) << "Implementation"
              << std::setw(30) << "Image"
              << std::setw(15) << "Encode Time (ms)"
              << std::setw(15) << "Decode Time (ms)"
              << std::setw(15) << "Encoded Size"
              << std::setw(10) << "Verified" << std::endl;

    for (const auto& impl : implementations) {
        for (const auto& image_path : image_paths) {
            int width, height, channels;
            unsigned char *img = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
            if (img == NULL) {
                std::cerr << "Error loading image: " << image_path << std::endl;
                continue;
            }
            std::vector<uint8_t> data(img, img + width * height * channels);
            stbi_image_free(img);

            QOIEncoderSpec enc_spec = {(uint32_t)width, (uint32_t)height, (uint8_t)channels, 0};
            std::vector<uint8_t> encoded_data;
            double encode_time = -1;
            if (impl.encoder) {
                auto start = std::chrono::high_resolution_clock::now();
                encoded_data = impl.encoder->encode(data, enc_spec);
                auto end = std::chrono::high_resolution_clock::now();
                encode_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            }

            QOIDecoderSpec dec_spec;
            std::vector<uint8_t> decoded_data;
            double decode_time = -1;
            bool verified = false;
            if (impl.decoder && !encoded_data.empty()) {
                auto start = std::chrono::high_resolution_clock::now();
                decoded_data = impl.decoder->decode(encoded_data, dec_spec);
                auto end = std::chrono::high_resolution_clock::now();
                decode_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                verified = (data == decoded_data);
            }

            std::cout << std::left << std::setw(20) << impl.name
                      << std::setw(30) << image_path.substr(image_path.find_last_of("/") + 1)
                      << std::setw(15) << (encode_time >= 0 ? std::to_string(encode_time) : "N/A")
                      << std::setw(15) << (decode_time >= 0 ? std::to_string(decode_time) : "N/A")
                      << std::setw(15) << (encoded_data.empty() ? "N/A" : std::to_string(encoded_data.size()))
                      << std::setw(10) << (verified ? "Yes" : "No") << std::endl;
        }
    }

    for (auto& impl : implementations) {
        delete impl.encoder;
        impl.encoder = nullptr;
        delete impl.decoder;
        impl.decoder = nullptr;
    }

    return 0;
}