#include "../reference/qoi-reference.hpp"
#include "qoi-mc.hpp"
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <algorithm>

int G_OUTER_THREADS = 3;
int G_INNER_THREADS = 0;
int G_MAX_ACTIVE_LEVELS = 2;
int G_DYNAMIC_THREADS = 0;

namespace fs = std::filesystem;

bool decode_qoi_to_raw(const fs::path &in_path,
                       std::vector<uint8_t> &pixels_out,
                       QOIDecoderSpec &spec_out);

int main(int argc, char **argv) {
  const int total_cores = omp_get_num_procs();

  omp_set_dynamic(G_DYNAMIC_THREADS);
  omp_set_max_active_levels(G_MAX_ACTIVE_LEVELS);
  omp_set_nested(G_MAX_ACTIVE_LEVELS > 1);

  int outer_threads = G_OUTER_THREADS > 0 ? G_OUTER_THREADS : total_cores;

  int inner_threads = G_INNER_THREADS > 0
                          ? G_INNER_THREADS
                          : std::max(1, total_cores / outer_threads);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " img1.qoi img2.qoi ...\n";
    return 1;
  }

  std::vector<fs::path> input_paths;
  for (int i = 1; i < argc; i++) {
    input_paths.emplace_back(argv[i]);
  }

  const int num_images = static_cast<int>(input_paths.size());
  omp_set_num_threads(inner_threads);


  double t_start = omp_get_wtime();

#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
  for (int i = 0; i < num_images; i++) {
    fs::path image_path = input_paths[i];
    std::vector<uint8_t> decoded_pixels;
    QOIDecoderSpec spec{};

    if (!decode_qoi_to_raw(image_path, decoded_pixels, spec)) {
      // decode failed → just skip this image
      continue;
    }
    QOIEncoderSpec enc_spec{};
    enc_spec.width = spec.width;
    enc_spec.height = spec.height;
    enc_spec.channels = spec.channels;
    enc_spec.colorspace = spec.colorspace;

    MultiCPUQOI encoder;
    std::vector<uint8_t> encoded = encoder.encode(decoded_pixels, enc_spec);

    QOIDecoderSpec mc_dec_spec{};
    std::vector<uint8_t> decoded_back = encoder.decode(encoded, mc_dec_spec);


  }
  double t_end = omp_get_wtime();
  std::cout << "Processed " << num_images << " images in " << (t_end - t_start)
            << " seconds (total)\n";

  return 0;
}