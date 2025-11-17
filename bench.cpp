#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <dirent.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "stb/stb_image.h"
#include "qoi.hpp"
#include "reference/qoi-reference.hpp"
#include "reference/libpng.hpp"
#include "reference/stb_image.hpp"
#include "single-cpu/qoi-sc.hpp"
#include "multi-cpu/qoi-mc.hpp"
#include "gpu-cuda/qoi-cude.hpp"
#include "stb/stb_image_write.h"

struct Implementation {
	std::string name;
	IEncoder* encoder;
	IDecoder* decoder;
};

struct ImageData {
	std::string path;
	std::vector<uint8_t> data;
	int width, height, channels, size;
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
		{"Single Threaded", new SingleCPUQOI(), new SingleCPUQOI()},
		{"Multi Threaded", new MultiCPUQOI(), new MultiCPUQOI()},
		{"GPU (CUDA)", new CUDAQOI(), new CUDAQOI()},
		// {"Libpng", new Libpng(), new Libpng()},
		// {"StbImage", new StbImage(), new StbImage()},
		// Comment to force formatting
	};

	std::vector<ImageData> images;
	DIR* dir;
	struct dirent* ent;
	if ((dir = opendir(images_dir.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			std::string filename = ent->d_name;
			if (filename.size() > 4 &&
				filename.substr(filename.size() - 4) == ".qoi") {
				ImageData img_data;
				img_data.path = images_dir + "/" + filename;
				img_data.size = std::filesystem::file_size(img_data.path);

				// Read whole .qoi file into memory
				std::ifstream in(img_data.path, std::ios::binary);
				if (!in) {
					std::cerr << "Error opening file: " << img_data.path
							  << std::endl;
					continue;
				}
				std::vector<uint8_t> file_bytes(
					(std::istreambuf_iterator<char>(in)),
					std::istreambuf_iterator<char>());

				// Decode QOI using the reference decoder to obtain raw pixels
				ReferenceQOI ref;
				QOIDecoderSpec spec{};
				auto pixels = ref.decode(file_bytes, spec);
				if (pixels.empty()) {
					std::cerr << "Error decoding QOI: " << img_data.path
							  << std::endl;
					continue;
				}

				img_data.width = spec.width;
				img_data.height = spec.height;
				img_data.channels = spec.channels;
				img_data.data = std::move(pixels);
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

	std::cout << std::left << std::setw(20) << "Implementation" << std::setw(40)
			  << "Image" << std::setw(20) << "Encode Time (ms)" << std::setw(20)
			  << "Decode Time (ms)" << std::setw(20) << "Encoding Ratio"
			  << std::setw(15) << "Verified" << std::endl;

	for (const auto& impl : implementations) {
		// Accumulate totals across all images and runs for this implementation
		double impl_total_encode_time_ms = 0.0;
		double impl_total_decode_time_ms = 0.0;
		uint64_t impl_total_raw_bytes_encoded =
			0;	// sum of width*height*channels per run
		uint64_t impl_total_pixels_encoded = 0;	 // sum of width*height per run

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
						// Output error image
						std::string out_path =
							image.path + "." + impl.name + ".err.qoi";
						ReferenceQOI ref;
						auto err_encoded = ref.encode(decoded_data, enc_spec);
						std::ofstream out(out_path, std::ios::binary);
						if (out) {
							out.write(reinterpret_cast<const char*>(
										  err_encoded.data()),
									  err_encoded.size());
						}
					}
				}
			}

			double size_ratio = encoded_data.empty()
									? 0.0
									: static_cast<double>(encoded_data.size()) /
										  static_cast<double>(image.size);
			std::cout << std::left << std::setw(20) << impl.name
					  << std::setw(40)
					  << image.path.substr(image.path.find_last_of("/") + 1)
					  << std::setw(20)
					  << (total_encode_time >= 0
							  ? std::to_string(total_encode_time / num_runs)
							  : "N/A")
					  << std::setw(20)
					  << (total_decode_time >= 0
							  ? std::to_string(total_decode_time / num_runs)
							  : "N/A")
					  << std::setw(20)
					  << (size_ratio > 0.0 ? "+" : "-") +
							 (encoded_data.empty()
								  ? "N/A"
								  : std::to_string((size_ratio - 1.0) *
												   100.0)) +
							 "%"
					  << std::setw(15) << (verified ? "Yes" : "No")
					  << std::endl;

			// update implementation-level totals
			// raw bytes (input pixels) per image per run
			uint64_t raw_bytes = static_cast<uint64_t>(image.width) *
								 static_cast<uint64_t>(image.height) *
								 static_cast<uint64_t>(image.channels);
			uint64_t pixels = static_cast<uint64_t>(image.width) *
							  static_cast<uint64_t>(image.height);
			impl_total_raw_bytes_encoded +=
				raw_bytes * static_cast<uint64_t>(num_runs);
			impl_total_pixels_encoded +=
				pixels * static_cast<uint64_t>(num_runs);
			impl_total_encode_time_ms += total_encode_time;
			impl_total_decode_time_ms += total_decode_time;
		}

		// After all images for this implementation, compute MB/s and Mp/s
		// (averaged over all pictures/runs)

		auto print_rate = [&](uint64_t total_raw_bytes, uint64_t total_pixels,
							  double total_time_ms, const std::string& label) {
			double total_time_s = total_time_ms / 1000.0;
			if (total_time_s <= 0.0) {
				std::cout << "  " << label << ": N/A (no time recorded)"
						  << std::endl;
				return;
			}
			double mb_per_s = static_cast<double>(total_raw_bytes) /
							  total_time_s / 1e6;  // MB/s (10^6)
			double mp_per_s = static_cast<double>(total_pixels) / total_time_s /
							  1e6;	// Mp/s (10^6)
			std::cout << "  " << label << ": " << std::fixed
					  << std::setprecision(2) << mb_per_s << " MB/s, "
					  << mp_per_s << " Mp/s" << std::endl;
		};

		std::cout << std::endl;
		std::cout << impl.name << " summary:" << std::endl;
		print_rate(impl_total_raw_bytes_encoded, impl_total_pixels_encoded,
				   impl_total_encode_time_ms, "Encode avg");
		print_rate(impl_total_raw_bytes_encoded, impl_total_pixels_encoded,
				   impl_total_decode_time_ms, "Decode avg");

		std::cout << std::endl;
	}

	for (auto& impl : implementations) {
		delete impl.encoder;
		impl.encoder = nullptr;
		delete impl.decoder;
		impl.decoder = nullptr;
	}

	return 0;
}
