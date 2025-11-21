#include "../reference/qoi-reference.hpp"
#include "qoi-mc.hpp"
#include "qoi-sc.hpp"
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

bool decode_qoi_to_raw(const fs::path &in_path,
                       std::vector<uint8_t> &pixels_out,
                       QOIDecoderSpec &spec_out);

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " img1.qoi img2.qoi ...\n";
    return 1;
  }

  std::vector<fs::path> input_paths;
  for (int i = 1; i < argc; i++) {
    input_paths.emplace_back(argv[i]);
  }

  const int num_images = static_cast<int>(input_paths.size());

  int outer_threads = omp_get_max_threads();
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

    SingleCPUQOI encoder;
    std::vector<uint8_t> encoded = encoder.encode(decoded_pixels, enc_spec);

    QOIDecoderSpec mc_dec_spec{};
    std::vector<uint8_t> decoded_back = encoder.decode(encoded, mc_dec_spec);

    // Output some info in case of debugging
#pragma omp critical
    {
      std::cout << "Thread " << omp_get_thread_num() << " decoded "
                << image_path << " (" << spec.width << "x" << spec.height
                << ", channels=" << int(spec.channels) << ")\n";
    }
  }

  double t_end = omp_get_wtime();
  std::cout << "Processed " << num_images << " images in " << (t_end - t_start)
            << " seconds (total)\n";

  return 0;
}
