#include "qoi-mc.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <omp.h>

#define QOI_OP_INDEX 0x00
#define QOI_OP_DIFF 0x40
#define QOI_OP_LUMA 0x80
#define QOI_OP_RUN 0xc0
#define QOI_OP_RGB 0xfe
#define QOI_OP_RGBA 0xff
#define QOI_MASK_2 0xc0

#define QOI_COLOR_HASH(C) \
	(C.rgba.r * 3 + C.rgba.g * 5 + C.rgba.b * 7 + C.rgba.a * 11)
#define QOI_MAGIC                                                \
	((((unsigned int)'q') << 24) | (((unsigned int)'o') << 16) | \
	 (((unsigned int)'i') << 8) | (((unsigned int)'f')))
#define QOI_HEADER_SIZE 14
#define QOI_PIXELS_MAX ((unsigned int)400000000)
#define CHECKPOINT_INTERVAL (1 << 20)  // 1 MiB
#define INVALID_PIXEL 0xFF'FF'FF'00	   // 0 alpha pure white

typedef union {
	struct {
		unsigned char r, g, b, a;
	} rgba;
	unsigned int v;
} qoi_rgba_t;

typedef union {
	struct {
		int32_t byte_offset;
		int32_t px_pos;
	} fields;
	char v[8];
} qoi_checkpoint_t;

static const unsigned char qoi_padding[8] = {0, 0, 0, 0, 0, 0, 0, 1};

static void qoi_write_32(unsigned char* bytes, int* p, unsigned int v) {
	bytes[(*p)++] = (0xff000000 & v) >> 24;
	bytes[(*p)++] = (0x00ff0000 & v) >> 16;
	bytes[(*p)++] = (0x0000ff00 & v) >> 8;
	bytes[(*p)++] = (0x000000ff & v);
}

static unsigned int qoi_read_32(const unsigned char* bytes, int* p) {
	unsigned int a = bytes[(*p)++];
	unsigned int b = bytes[(*p)++];
	unsigned int c = bytes[(*p)++];
	unsigned int d = bytes[(*p)++];
	return a << 24 | b << 16 | c << 8 | d;
}


std::vector<uint8_t> MultiCPUQOI::encode(const std::vector<uint8_t>& data,
										 QOIEncoderSpec& spec) {
	// int p = 0, run = 0;
	// int start_px;
	// int px_len, px_end, px_pos, channels;
	// std::vector<uint8_t> bytes;
	// std::vector<qoi_checkpoint_t> checkpoints;
	// const unsigned char* pixels = data.data();
	// qoi_rgba_t index[64];
	// qoi_rgba_t px, px_prev;

	if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}
	//header 

	// int max_size = spec.width * spec.height * (spec.channels + 1) +
	// 			   QOI_HEADER_SIZE + sizeof(qoi_padding);
	// int checkpoint_size =
	// 	(max_size / CHECKPOINT_INTERVAL) * sizeof(qoi_checkpoint_t) +
	// 	sizeof(qoi_padding);
	// bytes.resize(max_size + checkpoint_size);
	const unsigned char*  pixels = data.data();
    int channels = spec.channels;
	//size of image in pixels
    int total_px = spec.width * spec.height;

	std::vector<uint8_t> bytes;
	bytes.resize(QOI_HEADER_SIZE);
	int p = 0;
	qoi_write_32(bytes.data(), &p, QOI_MAGIC);
	qoi_write_32(bytes.data(), &p, spec.width);
	qoi_write_32(bytes.data(), &p, spec.height);
	//storing channels and colorspace in header
	bytes[p++] = spec.channels;
	bytes[p++] = spec.colorspace;

	// memset(index, INVALID_PIXEL, sizeof(index));

	

	// px_len = spec.width * spec.height * spec.channels;
	// px_end = px_len - spec.channels;
	// channels = spec.channels;

	// // Push first pixel to checkpoints
	// checkpoints.push_back((qoi_checkpoint_t){.fields = {p, start_px}});

	int max_threads = 1;
	#ifdef _OPENMP
    	max_threads = omp_get_max_threads();
	#endif
		// split image into segments for each thread
	const int chunk_px = 524288; // half a million pixels
	const int num_segments = std::max(1,std::min(max_threads, std::max(1, total_px / chunk_px))); // half a million pixels
	const int seg_px = (total_px + num_segments - 1) / num_segments;

	std::vector<std::vector<uint8_t>> bytes_parts(num_segments);
    std::vector<std::vector<qoi_checkpoint_t>> checkpoints_parts(num_segments);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(num_segments)
#endif


	for(int s = 0; s<num_segments;s++){
		//segment start pixel we do s*seg_px because s is segment index
		const int start_px = s * seg_px;
		//segment end pixel we use min to avoid overflow on last segment
		const int end_px = std::min(total_px, start_px + seg_px);
		if(start_px >= end_px) continue;

		std::vector<uint8_t>& bytes = bytes_parts[s];
		std::vector<qoi_checkpoint_t>& checkpoints = checkpoints_parts[s];

		//Local state in each thread
		int p = 0, run = 0;
		//px_len is number of pixels in this segment * channels
		//px_end is last pixel position in this segment
		//px_pos is current pixel position in this segment
		int px_len, px_end, px_pos, channels_local = channels; 
		qoi_rgba_t index[64]; //pixel color index table
		qoi_rgba_t px, px_prev;// current and previous pixel
		//seperate state dependent on segment
		memset(index, INVALID_PIXEL, sizeof(index));
		run = 0;
		px_prev.rgba.r = 0;
		px_prev.rgba.g = 0;
		px_prev.rgba.b = 0;
		px_prev.rgba.a = 255;
		px = px_prev;

		const int segment_start = start_px * channels_local;
		const int segment_end = end_px * channels_local;
		px_len = segment_end;
		px_end = segment_end - channels_local;

		bytes.reserve((end_px - start_px)*(channels_local +1));

        checkpoints.push_back((qoi_checkpoint_t){ .fields = { p, segment_start } });



	for (px_pos = segment_start; px_pos < segment_end; px_pos += channels_local) {
		//loads current pixel RGB
		px.rgba.r = pixels[px_pos + 0];
		px.rgba.g = pixels[px_pos + 1];
		px.rgba.b = pixels[px_pos + 2];

		//loads alpha of current pixel if exists
		if (channels == 4) {
			px.rgba.a = pixels[px_pos + 3];
		}

		//If current pixel is same as previous pixel
		if (px.v == px_prev.v) {
			// increase run length
			run++;
			///reach max encoded length push onto byte array and reset counter 
			if (run == 62 || px_pos == px_end) {
				bytes.push_back(QOI_OP_RUN | (run - 1));
				p++;
				run = 0;
			}
		} else {
			int index_pos;

			// if pixel color changed we push to byte and flush run 
			if (run > 0) {
				bytes.push_back( QOI_OP_RUN | (run - 1));
				p++;
				run = 0;
			}

			// Checkpoints are added so we can decode from other positions in
			// the middle of the image. This means we need to clear all state.
			// record a new checkpoint and reset encoder state (index + px_prev).
			if (p - checkpoints.back().fields.byte_offset >=
				CHECKPOINT_INTERVAL) {
				qoi_checkpoint_t cp = {.fields = {p, px_pos}};
				checkpoints.push_back(cp);
				memset(index, 0, sizeof(index)); //clear color index table
				px_prev.v = INVALID_PIXEL;  // force absolute write next time
			}

		// Hash current pixel to a color index slot [0..63].
		//basically maping the pixel to a position in the index table
			index_pos = QOI_COLOR_HASH(px) & (64 - 1); 

			// Fast path: if the hash slot holds the exact same pixel, emit INDEX op.
			if (index[index_pos].v != INVALID_PIXEL &&
				index[index_pos].v == px.v) {
				bytes.push_back( QOI_OP_INDEX | index_pos);
				p++;
			} else {
				// Otherwise, update the index slot with this pixel.
				index[index_pos] = px;

// If px_prev was invalidated (e.g., at start or after a checkpoint),
			// we must write this pixel absolutely (RGB/RGBA).
				if (px_prev.v == INVALID_PIXEL) {
					if (channels == 4) {
						bytes.push_back(QOI_OP_RGBA); 
						p++;
					} else {
						bytes.push_back(QOI_OP_RGB);
						p++;
					}
					bytes.push_back(px.rgba.r); p++;
					bytes.push_back(px.rgba.g); p++;
					bytes.push_back(px.rgba.b); p++;
			
					if (channels == 4) {
						bytes.push_back(px.rgba.a); p++;
					}
				} else {
					// If alpha is unchanged, try DIFF/LUMA (smaller) before falling back to RGB.
					if (px.rgba.a == px_prev.rgba.a) {
						signed char vr = px.rgba.r - px_prev.rgba.r;
						signed char vg = px.rgba.g - px_prev.rgba.g;
						signed char vb = px.rgba.b - px_prev.rgba.b;

						signed char vg_r = vr - vg;
						signed char vg_b = vb - vg;

						if (vr > -3 && vr < 2 && vg > -3 && vg < 2 && vb > -3 &&
							vb < 2) {
							bytes.push_back(QOI_OP_DIFF | (vr + 2) << 4 |
										 (vg + 2) << 2 | (vb + 2)); p++;
						} else if (vg_r > -9 && vg_r < 8 && vg > -33 &&
								   vg < 32 && vg_b > -9 && vg_b < 8) {
							bytes.push_back(QOI_OP_LUMA | (vg + 32)); 
							p++;
							bytes.push_back((vg_r + 8) << 4 | (vg_b + 8)); 
							p++;
						} else {
							bytes.push_back(QOI_OP_RGB); p++;
							bytes.push_back(px.rgba.r); p++;
							bytes.push_back(px.rgba.g); p++;
							bytes.push_back(px.rgba.b); p++;
						}
					} else {

						bytes.push_back(QOI_OP_RGBA); p++;
						bytes.push_back(px.rgba.r); p++;
						bytes.push_back(px.rgba.g); p ++;
						bytes.push_back(px.rgba.b); p++;
						bytes.push_back(px.rgba.a); p++;
					}
				}
			}
		}
		px_prev = px;
	}
	}

	//
	std::vector<size_t> segment_offsets(num_segments);
	size_t payload_bytes = 0;
	for (int i = 0; i < num_segments; i++) {
		segment_offsets[i] = payload_bytes;
		payload_bytes += bytes_parts[i].size();
		
	}
 bytes.reserve(QOI_HEADER_SIZE + payload_bytes +
                  sizeof(qoi_padding) +                   
				  (num_segments * sizeof(qoi_checkpoint_t)) +
                  sizeof(qoi_padding));

	for(int s = 0; s<num_segments;++s){
		bytes.insert(bytes.end(), bytes_parts[s].begin(), bytes_parts[s].end());
	}

	for (int i = 0; i < (int)sizeof(qoi_padding); ++i) bytes.push_back(qoi_padding[i]);


	// Add in second qoi ending padding
	  for (int s = 0; s < num_segments; ++s) {
        for (qoi_checkpoint_t cp : checkpoints_parts[s]) {
            cp.fields.byte_offset = QOI_HEADER_SIZE +
                                    (int32_t)segment_offsets[s] +
                                    cp.fields.byte_offset;
            for (uint8_t b : cp.v) bytes.push_back(b);
        }
    }

for (int i = 0; i < (int)sizeof(qoi_padding); ++i) bytes.push_back(qoi_padding[i]);

	return bytes;
}

std::vector<uint8_t> MultiCPUQOI::decode(
	const std::vector<uint8_t>& encoded_data,
	QOIDecoderSpec& spec) {
	// Read in header
	const unsigned char* bytes = encoded_data.data();
	int p = 0;
	unsigned int header_magic;

	if (encoded_data.size() < QOI_HEADER_SIZE + sizeof(qoi_padding)) {
		return {};
	}

	header_magic = qoi_read_32(bytes, &p);
	spec.width = qoi_read_32(bytes, &p);
	spec.height = qoi_read_32(bytes, &p);
	spec.channels = bytes[p++];
	spec.colorspace = bytes[p++];

	if (spec.width == 0 || spec.height == 0 || spec.channels < 3 ||
		spec.channels > 4 || spec.colorspace > 1 || header_magic != QOI_MAGIC ||
		spec.height >= QOI_PIXELS_MAX / spec.width) {
		return {};
	}

	int channels = spec.channels;
	int px_len = spec.width * spec.height * channels;

	// Read in checkpoints
	typedef union {
		struct {
			int32_t byte_offset;
			int32_t px_pos;
			int32_t next_px_pos;
		} fields;
		// Only 8, since next_px_pos is not stored in file
		char v[8];
	} checkpoint_span_t;
	std::vector<checkpoint_span_t> checkpoints;
	int max_size = spec.width * spec.height * (spec.channels + 1) +
				   QOI_HEADER_SIZE + sizeof(qoi_padding);
	int max_num_checkpoints = (max_size / CHECKPOINT_INTERVAL) + 1;
	int last_cp_px_pos = px_len;
	// Read backwards to check for double padding
	bool found_double_padding = false;
	int p_size = (int)sizeof(qoi_padding);
	int c_size = (int)sizeof(qoi_checkpoint_t);
	int cp_p = (int)encoded_data.size() - p_size;
	for (int i = 0; i < max_num_checkpoints; i++) {
		bool is_padding = true;
		for (int j = 0; j < (int)sizeof(qoi_padding); j++) {
			if (encoded_data[cp_p - p_size + j] != qoi_padding[j]) {
				is_padding = false;
				break;
			}
		}
		if (is_padding) {
			found_double_padding = true;
			// std::cout << "Found " << i << " checkpoints in QOI data."
			// 		  << std::endl;
			break;
		}
		checkpoint_span_t cp;
		for (int j = 0; j < c_size; j++) {
			cp.v[j] = encoded_data[cp_p - c_size + j];
		}
		// We're reading backwards, so this is right
		cp.fields.next_px_pos = last_cp_px_pos;
		last_cp_px_pos = cp.fields.px_pos;
		checkpoints.push_back(cp);
		cp_p -= sizeof(qoi_checkpoint_t);
	}
	if (!found_double_padding) {
		// No double padding found, reset to no checkpoints
		checkpoints.clear();
		checkpoints.push_back((checkpoint_span_t){.fields = {p, 0, px_len}});
		std::cout << "WARNING: No checkpoints found in QOI data, decoding from "
					 "beginning only."
				  << std::endl;
	}

	// Initialize stable state
	std::vector<uint8_t> pixels;
	pixels.resize(px_len);
	int chunks_len = encoded_data.size() - (int)sizeof(qoi_padding);

#pragma omp parallel for schedule(static)
	for (auto cp : checkpoints) {
		// std::cout << "Decoding from checkpoint at byte offset "
		// 		  << cp.fields.byte_offset << ", pixel position "
		// 		  << cp.fields.px_pos << std::endl;

		// Set per checkpoint variables
		int p_local = cp.fields.byte_offset;
		int px_pos = cp.fields.px_pos;

		// Initalize per checkpoint state
		qoi_rgba_t index[64];
		qoi_rgba_t px;
		int run = 0;

		memset(index, 0, sizeof(index));
		px.rgba.r = 0;
		px.rgba.g = 0;
		px.rgba.b = 0;
		px.rgba.a = 255;

		for (; px_pos < cp.fields.next_px_pos; px_pos += channels) {
			if (run > 0) {
				run--;
			} else if (p_local < chunks_len) {
				int b1 = bytes[p_local++];

				if (b1 == QOI_OP_RGB) {
					px.rgba.r = bytes[p_local++];
					px.rgba.g = bytes[p_local++];
					px.rgba.b = bytes[p_local++];
				} else if (b1 == QOI_OP_RGBA) {
					px.rgba.r = bytes[p_local++];
					px.rgba.g = bytes[p_local++];
					px.rgba.b = bytes[p_local++];
					px.rgba.a = bytes[p_local++];
				} else if ((b1 & QOI_MASK_2) == QOI_OP_INDEX) {
					px = index[b1];
				} else if ((b1 & QOI_MASK_2) == QOI_OP_DIFF) {
					px.rgba.r += ((b1 >> 4) & 0x03) - 2;
					px.rgba.g += ((b1 >> 2) & 0x03) - 2;
					px.rgba.b += (b1 & 0x03) - 2;
				} else if ((b1 & QOI_MASK_2) == QOI_OP_LUMA) {
					int b2 = bytes[p_local++];
					int vg = (b1 & 0x3f) - 32;
					px.rgba.r += vg - 8 + ((b2 >> 4) & 0x0f);
					px.rgba.g += vg;
					px.rgba.b += vg - 8 + (b2 & 0x0f);
				} else if ((b1 & QOI_MASK_2) == QOI_OP_RUN) {
					run = (b1 & 0x3f);
				}

				index[QOI_COLOR_HASH(px) & (64 - 1)] = px;
			}

			pixels[px_pos + 0] = px.rgba.r;
			pixels[px_pos + 1] = px.rgba.g;
			pixels[px_pos + 2] = px.rgba.b;

			if (channels == 4) {
				pixels[px_pos + 3] = px.rgba.a;
			}
		}
	}

	return pixels;
}
