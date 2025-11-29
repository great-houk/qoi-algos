#pragma once

#include "qoi.hpp"

class GPUQOI : public IEncoder, public IDecoder {
   private:
	int CHECKPOINT_INTERVAL;
	int CHECKPOINTS_PER_SEGMENT;

   public:
	GPUQOI(int checkpoint_interval = MIN_CHECKPOINT_INTERVAL,
				int checkpoints_per_segment = 4) {
		CHECKPOINT_INTERVAL = checkpoint_interval;
		CHECKPOINTS_PER_SEGMENT = checkpoints_per_segment;
	}
	~GPUQOI() override = default;

	std::vector<uint8_t> encode(const std::vector<uint8_t>& data,
								QOIEncoderSpec& spec) override;
	std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded_data,
								QOIDecoderSpec& spec) override;
};
