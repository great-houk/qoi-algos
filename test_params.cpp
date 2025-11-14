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
#include <thread>
#include <omp.h>

#include "qoi.hpp"
#include "multi-cpu/qoi-mc.hpp"
#include "reference/qoi-reference.hpp"

// This utility builds a parameter sweep for MultiCPUQOI constructor parameters
// and writes a CSV with columns:
// checkpoint_interval,checkpoints_per_segment,encode_MBps,encode_Mpps,decode_MBps,decode_Mpps
// Usage: test_params <num_runs> <images_dir> [out.csv]

struct ImageData {
	std::string path;
	std::vector<uint8_t> data;
	int width, height, channels, size;
};

static std::vector<ImageData> load_images(const std::string& images_dir) {
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
	}
	std::sort(
		images.begin(), images.end(),
		[](const ImageData& a, const ImageData& b) { return a.path < b.path; });
	return images;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0]
				  << " <num_runs> <images_dir> [out.csv]" << std::endl;
		return 1;
	}

	int num_runs = std::stoi(argv[1]);
	std::string images_dir = argv[2];
	std::string out_csv = (argc >= 4) ? argv[3] : "test_params.csv";

	// Read number of threads from environment (set by Slurm script).
	// Default to num cpu cores
	int num_threads = std::thread::hardware_concurrency();
	if (const char* nt = std::getenv("NUM_THREADS")) {
		try {
			num_threads = std::stoi(nt);
			if (num_threads < 1)
				num_threads = std::thread::hardware_concurrency();
		} catch (...) {
			num_threads = std::thread::hardware_concurrency();
		}
	}
	omp_set_num_threads(num_threads);

	auto images = load_images(images_dir);
	if (images.empty()) {
		std::cerr << "No images found in " << images_dir << std::endl;
		return 1;
	}

	// --- Adjustable parameter ranges ---
	// Change these vectors to adjust the sweep ranges.
	std::vector<int> checkpoint_intervals;
	for (int i = (1 << 15); i <= (1 << 24); i <<= 1) {
		checkpoint_intervals.push_back(i);
	}
	std::vector<int> checkpoints_per_segment = {2};

	// Open CSV. If it exists, append rows; otherwise create and write header
	// which includes `cpus`.
	bool file_exists = std::filesystem::exists(out_csv);
	std::ofstream out;
	if (file_exists) {
		out.open(out_csv, std::ios::app);
	} else {
		out.open(out_csv);
		if (!out) {
			std::cerr << "Failed to open output CSV: " << out_csv << std::endl;
			return 1;
		}
		out << "checkpoint_interval" << "," << "checkpoints_per_segment" << ","
			<< "cpus" << ",";
		out << "encode_MBps" << "," << "encode_Mpps" << ",";
		out << "decode_MBps" << "," << "decode_Mpps" << "\n";
	}

	// Sweep
	for (int ci : checkpoint_intervals) {
		for (int cps : checkpoints_per_segment) {
			// Create implementation instance for these parameters
			MultiCPUQOI impl(ci, cps);

			double total_encode_time_ms = 0.0;
			double total_decode_time_ms = 0.0;
			uint64_t total_raw_bytes = 0;  // sum over images and runs
			uint64_t total_pixels = 0;

			// For each image, run number of runs and sum times
			for (const auto& image : images) {
				std::vector<uint8_t> encoded_data;
				double image_total_encode_ms = 0.0;
				double image_total_decode_ms = 0.0;

				uint64_t raw_bytes = static_cast<uint64_t>(image.width) *
									 static_cast<uint64_t>(image.height) *
									 static_cast<uint64_t>(image.channels);
				uint64_t pixels = static_cast<uint64_t>(image.width) *
								  static_cast<uint64_t>(image.height);

				for (int run = 0; run < num_runs; ++run) {
					QOIEncoderSpec enc_spec = {(uint32_t)image.width,
											   (uint32_t)image.height,
											   (uint8_t)image.channels, 0};
					// encode
					auto start_e = std::chrono::high_resolution_clock::now();
					encoded_data = impl.encode(image.data, enc_spec);
					auto end_e = std::chrono::high_resolution_clock::now();
					double enc_ms =
						std::chrono::duration_cast<std::chrono::microseconds>(
							end_e - start_e)
							.count() /
						1000.0;
					image_total_encode_ms += enc_ms;

					// decode
					QOIDecoderSpec dec_spec{};
					auto start_d = std::chrono::high_resolution_clock::now();
					auto decoded = impl.decode(encoded_data, dec_spec);
					auto end_d = std::chrono::high_resolution_clock::now();
					double dec_ms =
						std::chrono::duration_cast<std::chrono::microseconds>(
							end_d - start_d)
							.count() /
						1000.0;
					image_total_decode_ms += dec_ms;

					// Optional: verify correctness
					if (decoded != image.data) {
						std::cerr << "Warning: decoded data mismatch for "
								  << image.path << " (ci=" << ci
								  << ", cps=" << cps << ")" << std::endl;
					}
				}

				// accumulate: multiply raw_bytes and pixels by number of runs
				total_raw_bytes += raw_bytes * static_cast<uint64_t>(num_runs);
				total_pixels += pixels * static_cast<uint64_t>(num_runs);
				total_encode_time_ms += image_total_encode_ms;
				total_decode_time_ms += image_total_decode_ms;
			}

			double total_encode_s = total_encode_time_ms / 1000.0;
			double total_decode_s = total_decode_time_ms / 1000.0;

			double encode_MBps = (total_encode_s > 0.0)
									 ? static_cast<double>(total_raw_bytes) /
										   total_encode_s / 1e6
									 : 0.0;
			double encode_Mpps =
				(total_encode_s > 0.0)
					? static_cast<double>(total_pixels) / total_encode_s / 1e6
					: 0.0;
			double decode_MBps = (total_decode_s > 0.0)
									 ? static_cast<double>(total_raw_bytes) /
										   total_decode_s / 1e6
									 : 0.0;
			double decode_Mpps =
				(total_decode_s > 0.0)
					? static_cast<double>(total_pixels) / total_decode_s / 1e6
					: 0.0;

			out << ci << "," << cps << "," << num_threads << ",";
			out << std::fixed << std::setprecision(2) << encode_MBps << ","
				<< encode_Mpps << ",";
			out << std::fixed << std::setprecision(2) << decode_MBps << ","
				<< decode_Mpps << "\n";

			std::cout << "ci=" << ci << " cps=" << cps
					  << " cpus=" << num_threads << " -> enc: " << encode_MBps
					  << " MB/s, " << encode_Mpps
					  << " Mp/s; dec: " << decode_MBps << " MB/s, "
					  << decode_Mpps << " Mp/s" << std::endl;
		}
	}

	out.close();
	std::cout << "Wrote " << out_csv << std::endl;
	return 0;
}
