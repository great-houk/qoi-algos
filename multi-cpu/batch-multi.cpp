#include "../reference/qoi-reference.hpp"
#include "qoi-mc.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <string>
#include <vector>

int G_OUTER_THREADS = 6;     // set to 0 to auto
int G_INNER_THREADS = 0;     // set to 0 to auto
int G_MAX_ACTIVE_LEVELS = 2; // nested parallelism depth
int G_DYNAMIC_THREADS = 0;  // 0 = false, 1 = true

namespace fs = std::filesystem;

static bool read_file_bytes(const fs::path &p, std::vector<uint8_t> &out) {
  std::ifstream in(p, std::ios::binary);
  if (!in)
    return false;
  out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

int main(int argc, char **argv) {
  const int total_cores = omp_get_num_procs();

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " img1.qoi img2.qoi ...\n";
    return 1;
  }

  // Collect input paths
  std::vector<fs::path> input_paths;
  input_paths.reserve(argc - 1);
  for (int i = 1; i < argc; i++) {
    input_paths.emplace_back(argv[i]);
  }

  // Stable ordering (helps comparisons)
  std::sort(input_paths.begin(), input_paths.end());

  const int num_images = static_cast<int>(input_paths.size());

  // OpenMP global settings
  omp_set_dynamic(G_DYNAMIC_THREADS);
  omp_set_max_active_levels(G_MAX_ACTIVE_LEVELS);
  omp_set_nested(G_MAX_ACTIVE_LEVELS > 1);

  // Compute outer threads, then clamp to sensible bounds
  int outer_threads = (G_OUTER_THREADS > 0) ? G_OUTER_THREADS : total_cores;
  if (outer_threads < 1)
    outer_threads = 1;
  outer_threads = std::min(outer_threads, total_cores);
  outer_threads = std::min(outer_threads, num_images);

  // Compute inner threads based on outer
  int inner_threads =
      (G_INNER_THREADS > 0) ? G_INNER_THREADS
                            : std::max(1, total_cores / outer_threads);

  // Set default thread count for inner parallel regions inside MultiCPUQOI
  omp_set_num_threads(inner_threads);

  // Print thread config up front (as requested)
  std::cout << "OpenMP config: "
            << "procs=" << total_cores
            << ", outer_threads=" << outer_threads
            << ", inner_threads=" << inner_threads
            << ", max_active_levels=" << G_MAX_ACTIVE_LEVELS
            << ", dynamic=" << (G_DYNAMIC_THREADS ? "true" : "false")
            << "\n";

  // ------------------------------------------------------------
  // PRELOAD PHASE (matches bench input provenance)
  // Read .qoi bytes and decode to raw pixels using ReferenceQOI
  // ------------------------------------------------------------
  std::vector<std::vector<uint8_t>> raw_pixels(num_images);
  std::vector<QOIEncoderSpec> enc_specs(num_images);

  for (int i = 0; i < num_images; i++) {
    const fs::path &image_path = input_paths[i];

    std::vector<uint8_t> file_bytes;
    if (!read_file_bytes(image_path, file_bytes)) {
      std::cerr << "Error opening file: " << image_path << "\n";
      continue;
    }

    ReferenceQOI ref;
    QOIDecoderSpec spec{};
    auto pixels = ref.decode(file_bytes, spec);
    if (pixels.empty()) {
      std::cerr << "Error decoding QOI with ReferenceQOI: " << image_path << "\n";
      continue;
    }

    raw_pixels[i] = std::move(pixels);

    QOIEncoderSpec enc{};
    enc.width = spec.width;
    enc.height = spec.height;
    enc.channels = spec.channels;
    enc.colorspace = spec.colorspace;
    enc_specs[i] = enc;
  }

  // Timers for MultiCPU nested encode/decode
  std::vector<double> mc_encode_times(num_images, 0.0);
  std::vector<double> mc_decode_times(num_images, 0.0);

  // ------------------------------------------------------------
  // TIMED NESTED PHASE
  // Outer parallel over images,
  // inner parallel inside MultiCPUQOI segments/checkpoints.
  // ------------------------------------------------------------
  double t_start = omp_get_wtime();

#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
  for (int i = 0; i < num_images; i++) {
    if (raw_pixels[i].empty())
      continue;

    MultiCPUQOI encoder;

    double t_enc_start = omp_get_wtime();
    std::vector<uint8_t> encoded = encoder.encode(raw_pixels[i], enc_specs[i]);
    double t_enc_end = omp_get_wtime();
    mc_encode_times[i] = t_enc_end - t_enc_start;

    QOIDecoderSpec mc_dec_spec{};
    double t_dec_start = omp_get_wtime();
    std::vector<uint8_t> decoded_back = encoder.decode(encoded, mc_dec_spec);
    double t_dec_end = omp_get_wtime();
    mc_decode_times[i] = t_dec_end - t_dec_start;

    // Optional strict correctness check (uncomment if you want noise-free runs)
    // if (decoded_back != raw_pixels[i]) {
    //   #pragma omp critical
    //   std::cerr << "Mismatch after roundtrip: " << input_paths[i] << "\n";
    // }
  }

  double t_end = omp_get_wtime();

  std::cout << "Processed " << num_images << " images in "
            << (t_end - t_start) << " seconds (total)\n";

  for (int i = 0; i < num_images; ++i) {
    std::string name = input_paths[i].filename().string();
    std::cout << name
              << "  encode=" << mc_encode_times[i]
              << " s, decode=" << mc_decode_times[i]
              << " s\n";
  }

  return 0;
}
