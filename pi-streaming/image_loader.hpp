#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>

namespace fs = std::filesystem;

struct ImageFile {
    std::string filename;                  // full path or just name
    std::vector<std::uint8_t> data;        // raw bytes of the file
};

extern std::vector<ImageFile> images;

// Read a single file into a byte vector
std::vector<std::uint8_t> readFileToBytes(const fs::path &filePath);

// Load images from a folder into a vector
std::vector<ImageFile> loadImagesFromDirectory(
    const std::string &directoryPath,
    const std::string &extensionFilter = ""
);

// Initialize global images vector
void initImages();

#endif // IMAGE_LOADER_HPP
