#include "image_loader.hpp"
#include <fstream>

std::vector<ImageFile> images;

// Read a single file into a byte vector
std::vector<std::uint8_t> readFileToBytes(const fs::path &filePath) {
    std::vector<std::uint8_t> buffer;

    // Open in binary mode at end to get size quickly
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filePath.string());
    }

    std::streamsize size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to get size for file: " + filePath.string());
    }

    buffer.resize(static_cast<std::size_t>(size));

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filePath.string());
    }

    return buffer;
}

// Load images from directory
std::vector<ImageFile> loadImagesFromDirectory(
    const std::string &directoryPath,
    const std::string &extensionFilter
) {
    std::vector<ImageFile> images;

    fs::path dir(directoryPath);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("Invalid directory: " + directoryPath);
    }

    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path &p = entry.path();
        if (!extensionFilter.empty() && p.extension() != extensionFilter) {
            continue;
        }

        ImageFile img;
        img.filename = p.string();
        img.data = readFileToBytes(p);
        images.push_back(std::move(img));
    }

    return images;
}

void initImages() {
    static bool inited = false;
    if (inited) return;

    try {
        images = loadImagesFromDirectory("test_photos", ".qoi");
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
    }

    inited = true;
}
