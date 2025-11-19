#include "../reference/qoi-reference.hpp"
#include "qoi-mc.hpp"
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

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_images; i++) {
    fs::path image_path = input_paths[i];
    std::vector<uint8_t> decoded_pixels;
    QOIDecoderSpec spec{};

    if (!decode_qoi_to_raw(image_path, decoded_pixels, spec)) {
      // decode failed → just skip this image
      continue;
    }

    encode(decoded_pixels, spec);
    // Output some info in case of debugging
   #pragma omp critical
    {
      std::cout << "Thread " << omp_get_thread_num()
                << " decoded " << image_path
                << " (" << spec.width << "x" << spec.height
                << ", channels=" << int(spec.channels) << ")\n";
    }


    

  }

  return 0;
}