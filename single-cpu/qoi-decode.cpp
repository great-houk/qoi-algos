#include "../reference/qoi-reference.hpp"
#include "../qoi.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>


namespace fs = std::filesystem;

static std::vector<uint8_t> read_file_bytes(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {};
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

static bool write_file_bytes(const std::string &path,
                             const std::vector<uint8_t> &data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char *>(data.data()), data.size());
  return true;
}

bool decode_qoi_to_raw(const fs::path &in_path,
                       std::vector<uint8_t> &pixels_out,
                       QOIDecoderSpec &spec_out) {
    auto encoded = read_file_bytes(in_path);
    if (encoded.empty()) {
        std::cerr << "Failed to read input file: " << in_path << "\n";
        return false;
    }

    ReferenceQOI ref;
    QOIDecoderSpec spec{};
    auto pixels = ref.decode(encoded, spec);
    if (pixels.empty()) {
        std::cerr << "QOI decode failed for: " << in_path << "\n";
        return false;
    }

    pixels_out = std::move(pixels);
    spec_out = spec;
    return true;
}
