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

struct ImageData {
	std::string path;
	std::vector<uint8_t> data;
	int width, height, channels;
};

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <num_runs> <images_directory>"
				  << std::endl;
		return 1;
	}

	int num_runs = std::stoi(argv[1]);
	std::string images_dir = argv[2];

	std::vector<Implementation> implementations = {
		{"Reference", new ReferenceQOI(), new ReferenceQOI()},
		{"Single CPU", new SingleCPUQOI(), new SingleCPUQOI()}};

	std::vector<ImageData> images;
	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(images_dir.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			std::string filename = ent->d_name;
			if (filename.size() > 4 &&
				filename.substr(filename.size() - 4) == ".png") {
				ImageData img_data;
				img_data.path = images_dir + "/" + filename;
				unsigned char* img =
					stbi_load(img_data.path.c_str(), &img_data.width,
							  &img_data.height, &img_data.channels, 0);
				if (img == NULL) {
					std::cerr << "Error loading image: " << img_data.path
							  << std::endl;
					continue;
				}
				img_data.data.assign(
					img,
					img + img_data.width * img_data.height * img_data.channels);
				stbi_image_free(img);
				images.push_back(img_data);
			}
		}
		closedir(dir);
	} else {
		std::cerr << "Error opening directory: " << images_dir << std::endl;
		return 1;
	}
	std::sort(
		images.begin(), images.end(),
		[](const ImageData& a, const ImageData& b) { return a.path < b.path; });

	std::cout << std::left << std::setw(20) << "Implementation" << std::setw(30)
			  << "Image" << std::setw(15) << "Encode Time (ms)" << std::setw(15)
			  << "Decode Time (ms)" << std::setw(15) << "Encoded Size"
			  << std::setw(10) << "Verified" << std::endl;

	for (const auto& impl : implementations) {
		for (const auto& image : images) {
			double total_encode_time = 0;
			double total_decode_time = 0;
			std::vector<uint8_t> encoded_data;
			bool verified = true;

			for (int run = 0; run < num_runs; ++run) {
				QOIEncoderSpec enc_spec = {(uint32_t)image.width,
										   (uint32_t)image.height,
										   (uint8_t)image.channels, 0};
				double encode_time = -1;
				if (impl.encoder) {
					auto start = std::chrono::high_resolution_clock::now();
					encoded_data = impl.encoder->encode(image.data, enc_spec);
					auto end = std::chrono::high_resolution_clock::now();
					encode_time =
						std::chrono::duration_cast<std::chrono::microseconds>(
							end - start)
							.count() /
						1000.0;
					total_encode_time += encode_time;
				}

				QOIDecoderSpec dec_spec;
				std::vector<uint8_t> decoded_data;
				double decode_time = -1;
				if (impl.decoder && !encoded_data.empty()) {
					auto start = std::chrono::high_resolution_clock::now();
					decoded_data = impl.decoder->decode(encoded_data, dec_spec);
					auto end = std::chrono::high_resolution_clock::now();
					decode_time =
						std::chrono::duration_cast<std::chrono::microseconds>(
							end - start)
							.count() /
						1000.0;
					total_decode_time += decode_time;
					if (image.data != decoded_data) {
						verified = false;
					}
				}
			}

			std::cout << std::left << std::setw(20) << impl.name
					  << std::setw(30)
					  << image.path.substr(image.path.find_last_of("/") + 1)
					  << std::setw(15)
					  << (total_encode_time >= 0
							  ? std::to_string(total_encode_time / num_runs)
							  : "N/A")
					  << std::setw(15)
					  << (total_decode_time >= 0
							  ? std::to_string(total_decode_time / num_runs)
							  : "N/A")
					  << std::setw(15)
					  << (encoded_data.empty()
							  ? "N/A"
							  : std::to_string(encoded_data.size()))
					  << std::setw(10) << (verified ? "Yes" : "No")
					  << std::endl;
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