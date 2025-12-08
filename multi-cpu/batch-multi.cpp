#include "../reference/qoi-reference.hpp"
#include "qoi-mc.hpp"
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

int G_OUTER_THREADS = 2;
int G_INNER_THREADS = 0;
int G_MAX_ACTIVE_LEVELS = 2;
int G_DYNAMIC_THREADS = 0;

namespace fs = std::filesystem;

bool decode_qoi_to_raw(const fs::path& in_path,
					   std::vector<uint8_t>& pixels_out,
					   QOIDecoderSpec& spec_out);

struct LoadedImage {
	fs::path path;
	std::vector<uint8_t> pixels;  // raw pixels (decoded from qoi)
	QOIDecoderSpec spec;
	size_t raw_bytes() const { return pixels.size(); }
};

static void print_usage(const char* prog) {
	std::cerr << "Usage: " << prog
			  << " [--outer N] [--inner M] [--dynamic 0|1] "
				 "[--max-active-levels N] img1.qoi img2.qoi ...\n";
}

int main(int argc, char** argv) {
	const int total_cores = omp_get_num_procs();

	// simple CLI parsing for thread counts
	int outer_threads = G_OUTER_THREADS;
	int inner_threads = G_INNER_THREADS;
	int dynamic_threads = G_DYNAMIC_THREADS;
	int max_active = G_MAX_ACTIVE_LEVELS;

	std::vector<fs::path> input_paths;
	for (int i = 1; i < argc; ++i) {
		std::string s = argv[i];
		if (s == "--outer" || s == "-o") {
			if (i + 1 >= argc) {
				print_usage(argv[0]);
				return 1;
			}
			outer_threads = std::stoi(argv[++i]);
		} else if (s == "--inner" || s == "-i") {
			if (i + 1 >= argc) {
				print_usage(argv[0]);
				return 1;
			}
			inner_threads = std::stoi(argv[++i]);
		} else if (s == "--dynamic") {
			if (i + 1 >= argc) {
				print_usage(argv[0]);
				return 1;
			}
			dynamic_threads = std::stoi(argv[++i]);
		} else if (s == "--max-active-levels") {
			if (i + 1 >= argc) {
				print_usage(argv[0]);
				return 1;
			}
			max_active = std::stoi(argv[++i]);
		} else if (s == "--help" || s == "-h") {
			print_usage(argv[0]);
			return 0;
		} else {
			input_paths.emplace_back(s);
		}
	}

	if (input_paths.empty()) {
		print_usage(argv[0]);
		return 1;
	}

	omp_set_dynamic(dynamic_threads);
	omp_set_max_active_levels(max_active);
	omp_set_nested(max_active > 1);

	if (outer_threads <= 0)
		outer_threads = std::max(1, total_cores);
	if (inner_threads <= 0)
		inner_threads = std::max(1, total_cores / outer_threads);

	std::cout << "Config: outer_threads=" << outer_threads
			  << " inner_threads=" << inner_threads
			  << " dynamic=" << dynamic_threads
			  << " max_active_levels=" << max_active << "\n";

	// Load (decode) all files before timing the encode/decode pipeline.
	std::vector<LoadedImage> images;
	images.reserve(input_paths.size());
	for (const auto& p : input_paths) {
		LoadedImage li;
		li.path = p;
		if (!decode_qoi_to_raw(p, li.pixels, li.spec)) {
			std::cerr << "Failed to decode " << p << " — skipping\n";
			continue;
		}
		images.push_back(std::move(li));
	}

	const int num_images = static_cast<int>(images.size());
	if (num_images == 0) {
		std::cerr << "No images decoded — exiting\n";
		return 1;
	}

	std::vector<double> encode_times(num_images, 0.0);
	std::vector<double> decode_times(num_images, 0.0);
	std::vector<size_t> raw_bytes(num_images, 0);

	// Compute raw bytes per image (before timing)
	for (int i = 0; i < num_images; ++i)
		raw_bytes[i] = images[i].raw_bytes();

	// Run the pipeline: parallelize over images (outer), allow inner threading
	omp_set_num_threads(inner_threads);
	double t_start = omp_get_wtime();

#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for (int i = 0; i < num_images; ++i) {
		const LoadedImage& li = images[i];

		QOIEncoderSpec enc_spec{};
		enc_spec.width = li.spec.width;
		enc_spec.height = li.spec.height;
		enc_spec.channels = li.spec.channels;
		enc_spec.colorspace = li.spec.colorspace;

		MultiCPUQOI encoder;

		double t_enc0 = omp_get_wtime();
		std::vector<uint8_t> encoded = encoder.encode(li.pixels, enc_spec);
		double t_enc1 = omp_get_wtime();
		encode_times[i] = t_enc1 - t_enc0;

		QOIDecoderSpec dec_spec{};
		double t_dec0 = omp_get_wtime();
		std::vector<uint8_t> decoded_back = encoder.decode(encoded, dec_spec);
		double t_dec1 = omp_get_wtime();
		decode_times[i] = t_dec1 - t_dec0;
	}

	double t_end = omp_get_wtime();
	double total_seconds = t_end - t_start;

	size_t total_raw =
		std::accumulate(raw_bytes.begin(), raw_bytes.end(), size_t(0));
	// We did an encode (input raw bytes) and a decode (output raw bytes) per
	// image
	double bytes_processed = static_cast<double>(total_raw) * 2.0;
	double mb_processed = bytes_processed / (1024.0 * 1024.0);
	double mb_per_sec = mb_processed / total_seconds;

	std::cout << "Processed " << num_images << " images in " << total_seconds
			  << " seconds (wall)\n";
	std::cout << "Total raw bytes (sum of images) = " << total_raw
			  << " bytes\n";
	std::cout << "Total MB processed (encode+decode) = " << mb_processed
			  << " MB\n";
	std::cout << "Pipeline throughput = " << mb_per_sec << " MB/s\n";

	// Per-image breakdown
	for (int i = 0; i < num_images; ++i) {
		std::string name = images[i].path.filename().string();
		double mb = static_cast<double>(raw_bytes[i]) / (1024.0 * 1024.0);
		double enc_mb_s = mb / encode_times[i];
		double dec_mb_s = mb / decode_times[i];
		std::cout << name << "  size=" << mb
				  << " MB, encode=" << encode_times[i] << " s (" << enc_mb_s
				  << " MB/s) , decode=" << decode_times[i] << " s (" << dec_mb_s
				  << " MB/s)\n";
	}

	return 0;
}
