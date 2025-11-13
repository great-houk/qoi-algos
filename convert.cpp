#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "reference/libpng.hpp"
#include "reference/qoi-reference.hpp"
#include "qoi.hpp"

namespace fs = std::filesystem;

static std::vector<uint8_t> read_file_bytes(const fs::path& p) {
	std::ifstream in(p, std::ios::binary);
	if (!in)
		return {};
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

static bool write_file_bytes(const fs::path& p,
							 const std::vector<uint8_t>& data) {
	std::ofstream out(p, std::ios::binary);
	if (!out)
		return false;
	out.write(reinterpret_cast<const char*>(data.data()), data.size());
	return true;
}

int main(int argc, char** argv) {
	std::vector<fs::path> dirs = {"images", "images_small"};
	if (argc > 1) {
		// allow overriding or adding directories from CLI
		dirs.clear();
		for (int i = 1; i < argc; ++i)
			dirs.emplace_back(argv[i]);
	}

	ReferenceQOI encoder;
	Libpng png;

	for (const auto& dir : dirs) {
		if (!fs::exists(dir) || !fs::is_directory(dir)) {
			std::cerr << "Skipping missing or non-directory: " << dir << '\n';
			continue;
		}

		for (auto& entry : fs::directory_iterator(dir)) {
			if (!entry.is_regular_file())
				continue;
			auto ext = entry.path().extension().string();
			for (auto& c : ext)
				c = (char)tolower((unsigned char)c);
			if (ext != ".png")
				continue;

			std::cout << "Converting: " << entry.path() << " -> ";
			try {
				auto in_bytes = read_file_bytes(entry.path());
				if (in_bytes.empty()) {
					std::cerr << "failed to read file\n";
					continue;
				}

				QOIDecoderSpec dspec{};
				auto pixels = png.decode(in_bytes, dspec);
				if (pixels.empty()) {
					std::cerr << "png decode failed\n";
					continue;
				}

				QOIEncoderSpec espec{};
				espec.width = dspec.width;
				espec.height = dspec.height;
				// libpng decode forces 4 channels in this repo's implementation
				espec.channels = (dspec.channels >= 3 && dspec.channels <= 4)
									 ? dspec.channels
									 : 4;
				espec.colorspace = 0;

				auto out_bytes = encoder.encode(pixels, espec);
				if (out_bytes.empty()) {
					std::cerr << "qoi encode failed\n";
					continue;
				}

				fs::path out_path = entry.path();
				out_path.replace_extension(".qoi");
				if (!write_file_bytes(out_path, out_bytes)) {
					std::cerr << "write failed\n";
					continue;
				}

				std::cout << out_path << '\n';
			} catch (const std::exception& e) {
				std::cerr << "error: " << e.what() << '\n';
			}
		}
	}

	return 0;
}
