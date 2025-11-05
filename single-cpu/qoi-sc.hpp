#pragma once

#include "qoi.hpp"

class SingleCPUQOI : public IEncoder, public IDecoder {
public:
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data, QOIEncoderSpec& spec) override;
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded_data, QOIDecoderSpec& spec) override;
};
