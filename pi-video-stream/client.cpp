// ---- Platform-specific networking includes ----
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

// ---- Standard includes ----
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cerrno>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/libcamera.h>
#include <libcamera/stream.h>

#include "../multi-cpu/qoi-mc.hpp"

#define PORT 8080

// ---- Small cross-platform helpers ----
#ifdef _WIN32
using socket_t = SOCKET;
static bool init_sockets() {
	WSADATA wsa;
	return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}
static void cleanup_sockets() {
	WSACleanup();
}
static void close_socket(socket_t s) {
	closesocket(s);
}
#else
using socket_t = int;
static bool init_sockets() {
	return true;
}
static void cleanup_sockets() {}
static void close_socket(socket_t s) {
	close(s);
}
#endif

static bool send_all(socket_t s, const uint8_t* data, size_t len) {
	size_t total_sent = 0;
	while (total_sent < len) {
#ifdef _WIN32
		int sent = send(s, reinterpret_cast<const char*>(data + total_sent),
						static_cast<int>(len - total_sent), 0);
		if (sent <= 0)
			return false;
#else
		ssize_t sent = send(s, reinterpret_cast<const char*>(data + total_sent),
							len - total_sent, 0);
		if (sent <= 0)
			return false;
#endif
		total_sent += static_cast<size_t>(sent);
	}
	return true;
}

int main() {
	if (!init_sockets()) {
		std::cerr << "Failed to initialize socket subsystem" << std::endl;
		return -1;
	}

	socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (server_fd < 0) {
		std::cerr << "Socket creation error" << std::endl;
		cleanup_sockets();
		return -1;
	}

	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(PORT);

	if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
		std::cerr << "Bind failed" << std::endl;
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	if (listen(server_fd, 1) < 0) {
		std::cerr << "Listen failed" << std::endl;
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	std::cout << "Waiting for a viewer to connect on port " << PORT << "..."
			  << std::endl;
	socket_t client = accept(server_fd, nullptr, nullptr);
	if (client < 0) {
		std::cerr << "Accept failed" << std::endl;
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	// --- libcamera setup ---
	libcamera::CameraManager cm;
	if (cm.start()) {
		std::cerr << "Failed to start CameraManager" << std::endl;
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}
	if (cm.cameras().empty()) {
		std::cerr << "No cameras found" << std::endl;
		cm.stop();
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	std::shared_ptr<libcamera::Camera> camera = cm.cameras()[0];
	if (!camera) {
		std::cerr << "Failed to get camera" << std::endl;
		cm.stop();
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	if (camera->acquire()) {
		std::cerr << "Failed to acquire camera" << std::endl;
		camera.reset();
		cm.stop();
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	std::unique_ptr<libcamera::CameraConfiguration> config =
		camera->generateConfiguration({libcamera::StreamRole::VideoRecording});
	if (!config || config->size() == 0) {
		std::cerr << "Failed to generate configuration" << std::endl;
		camera->release();
		camera.reset();
		cm.stop();
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	libcamera::StreamConfiguration& cfg = config->at(0);
	const libcamera::StreamFormats& formats = cfg.formats();

	// Prefer RGB888 if available; otherwise fall back to the first reported format.
	libcamera::PixelFormat chosen_pf = libcamera::formats::RGB888;
	std::vector<libcamera::PixelFormat> pixel_formats = formats.pixelformats();
	if (std::find(pixel_formats.begin(), pixel_formats.end(), chosen_pf) ==
		pixel_formats.end()) {
		if (!pixel_formats.empty()) {
			chosen_pf = pixel_formats.front();
		}
	}
	cfg.pixelFormat = chosen_pf;

	// Pick the smallest available size for the chosen format to minimize bandwidth.
	std::vector<libcamera::Size> sizes = formats.sizes(cfg.pixelFormat);
	if (!sizes.empty()) {
		libcamera::Size min_size = sizes.front();
		for (const auto& s : sizes) {
			if (s.width * s.height < min_size.width * min_size.height) {
				min_size = s;
			}
		}
		cfg.size = min_size;
	} else {
		cfg.size.width = 640;
		cfg.size.height = 480;
	}
	cfg.bufferCount = 4;
	config->validate();

	if (camera->configure(config.get())) {
		std::cerr << "Failed to configure camera" << std::endl;
		camera->release();
		camera.reset();
		cm.stop();
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	libcamera::FrameBufferAllocator allocator(camera);
	libcamera::Stream* stream = cfg.stream();
	if (allocator.allocate(stream) < 0) {
		std::cerr << "Failed to allocate buffers" << std::endl;
		camera->release();
		camera.reset();
		cm.stop();
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	// Map buffers
	struct MappedBuffer {
		libcamera::FrameBuffer* fb;
		std::vector<void*> map_bases;
		std::vector<void*> data_ptrs;
		std::vector<size_t> map_lengths;
	};
	std::map<libcamera::FrameBuffer*, MappedBuffer> buffer_map;

	for (const std::unique_ptr<libcamera::FrameBuffer>& fb :
		 allocator.buffers(stream)) {
		MappedBuffer mb{fb.get(), {}, {}, {}};
		for (const auto& plane : fb->planes()) {
			size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
			if (page == 0)
				page = 4096;
			const size_t aligned_offset =
				plane.offset & ~(page - 1);
			const size_t offset_delta = plane.offset - aligned_offset;
			const size_t map_length = plane.length + offset_delta;

			void* map_base = mmap(nullptr, map_length, PROT_READ, MAP_SHARED,
								  plane.fd.get(), aligned_offset);
			if (map_base == MAP_FAILED) {
				int err = errno;
				std::cerr << "mmap failed (read-only), errno=" << err << " ("
						  << std::strerror(err) << "), retrying RW" << std::endl;
				map_base = mmap(nullptr, map_length, PROT_READ | PROT_WRITE,
								MAP_SHARED, plane.fd.get(), aligned_offset);
			}
			if (map_base == MAP_FAILED) {
				int err = errno;
				std::cerr << "mmap failed even with PROT_WRITE, errno=" << err
						  << " (" << std::strerror(err) << ")" << std::endl;
				camera->release();
				camera.reset();
				cm.stop();
				close_socket(client);
				close_socket(server_fd);
				cleanup_sockets();
				return -1;
			}

			uint8_t* data_ptr =
				static_cast<uint8_t*>(map_base) + static_cast<std::ptrdiff_t>(offset_delta);
			mb.map_bases.push_back(map_base);
			mb.data_ptrs.push_back(data_ptr);
			mb.map_lengths.push_back(map_length);
		}
		buffer_map[fb.get()] = std::move(mb);
	}

	struct Frame {
		std::vector<uint8_t> pixels;  // RGB bytes
		uint32_t width;
		uint32_t height;
		std::chrono::steady_clock::time_point captured;
	};

	std::deque<Frame> queue;
	std::mutex mtx;
	std::condition_variable cv_not_full;
	std::condition_variable cv_not_empty;
	const size_t kMaxQueue = 3;	 // keep a small buffer to avoid lag
	std::atomic<bool> running{true};

	MultiCPUQOI encoder;

	struct RequestHandler {
		libcamera::Camera* cam;
		libcamera::StreamConfiguration* cfg;
		std::map<libcamera::FrameBuffer*, MappedBuffer>* buffer_map;
		std::deque<Frame>* queue;
		std::mutex* mtx;
		std::condition_variable* cv_not_full;
		std::condition_variable* cv_not_empty;
		const size_t* kMaxQueue;
		std::atomic<bool>* running;

		void on_request_complete(libcamera::Request* request) {
			if (!running->load())
				return;
			if (request->status() == libcamera::Request::RequestCancelled)
				return;

			Frame f;
			f.width = static_cast<uint32_t>(cfg->size.width);
			f.height = static_cast<uint32_t>(cfg->size.height);
			f.pixels.resize(static_cast<size_t>(f.width) * f.height * 3);
			f.captured = std::chrono::steady_clock::now();

			for (auto& [stream_ptr, buffer] : request->buffers()) {
				(void)stream_ptr;
				auto it = buffer_map->find(buffer);
				if (it == buffer_map->end())
					continue;
				MappedBuffer& mb = it->second;
				const libcamera::FrameMetadata& md = buffer->metadata();
				size_t bytes_used =
					md.planes().empty() ? 0 : md.planes()[0].bytesused;
				if (bytes_used == 0)
					bytes_used = f.pixels.size();
				std::memcpy(f.pixels.data(), mb.data_ptrs[0],
							std::min(bytes_used, f.pixels.size()));
			}

			{
				std::unique_lock<std::mutex> lock(*mtx);
				cv_not_full->wait(
					lock, [&] { return !running->load() || queue->size() < *kMaxQueue; });
				if (!running->load())
					return;
				queue->push_back(std::move(f));
				cv_not_empty->notify_one();
			}

			request->reuse(libcamera::Request::ReuseBuffers);
			cam->queueRequest(request);
		}
	};

	RequestHandler handler{camera.get(),
						   &cfg,
						   &buffer_map,
						   &queue,
						   &mtx,
						   &cv_not_full,
						   &cv_not_empty,
						   &kMaxQueue,
						   &running};
	camera->requestCompleted.connect(&handler, &RequestHandler::on_request_complete);

	std::vector<std::unique_ptr<libcamera::Request>> requests;
	for (const std::unique_ptr<libcamera::FrameBuffer>& fb :
		 allocator.buffers(stream)) {
		std::unique_ptr<libcamera::Request> req = camera->createRequest();
		if (!req) {
			std::cerr << "Failed to create request" << std::endl;
			running = false;
			break;
		}
		if (req->addBuffer(stream, fb.get())) {
			std::cerr << "Failed to add buffer to request" << std::endl;
			running = false;
			break;
		}
			// Set frame duration limits to target ~10 fps.
			int64_t frame_time = 1000000000LL / 10;
		req->controls().set(
			libcamera::controls::FrameDurationLimits,
			libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
		requests.push_back(std::move(req));
	}

	if (running && camera->start()) {
		std::cerr << "Failed to start camera" << std::endl;
		running = false;
	}

	if (running) {
		for (auto& req : requests) {
			if (camera->queueRequest(req.get())) {
				std::cerr << "Failed to queue request" << std::endl;
				running = false;
				break;
			}
		}
		}

		// Encode/send thread: pops frames, encodes with multi-core QOI, sends.
		std::thread encode_thread([&]() {
			while (running) {
				Frame f;
				{
					std::unique_lock<std::mutex> lock(mtx);
					cv_not_empty.wait(lock,
									  [&] { return !running || !queue.empty(); });
					if (!running && queue.empty())
						break;
					f = std::move(queue.front());
					queue.pop_front();
					cv_not_full.notify_one();
				}

				QOIEncoderSpec spec{
					f.width,
					f.height,
					3,
					0,
				};

				auto encode_start = std::chrono::steady_clock::now();
				std::vector<uint8_t> encoded = encoder.encode(f.pixels, spec);
				auto encode_end = std::chrono::steady_clock::now();
				double encode_ms =
					std::chrono::duration<double, std::milli>(encode_end -
															  encode_start)
						.count();

				auto send_start = std::chrono::steady_clock::now();
				uint32_t payload_size = static_cast<uint32_t>(encoded.size());
				uint32_t net_size = htonl(payload_size);

				if (!send_all(client, reinterpret_cast<uint8_t*>(&net_size),
							  sizeof(net_size))) {
					std::cerr << "Failed to send frame length" << std::endl;
					running = false;
					cv_not_empty.notify_all();
					break;
				}

				if (!send_all(client, encoded.data(), encoded.size())) {
					std::cerr << "Failed to send frame payload" << std::endl;
					running = false;
					cv_not_empty.notify_all();
					break;
				}
				auto send_end = std::chrono::steady_clock::now();

				double capture_to_encode_ms =
					std::chrono::duration<double, std::milli>(encode_start - f.captured)
						.count();
				double send_ms =
					std::chrono::duration<double, std::milli>(send_end - send_start)
						.count();
				double capture_to_send_ms =
					std::chrono::duration<double, std::milli>(send_end - f.captured)
						.count();
				std::cout << "Timing: capture->encode_start=" << capture_to_encode_ms
						  << " ms, encode=" << encode_ms << " ms, send=" << send_ms
						  << " ms, capture->send=" << capture_to_send_ms << " ms"
						  << std::endl;
			}
		});

	std::cout << "Streaming frames..." << std::endl;

	encode_thread.join();
	running = false;
	cv_not_full.notify_all();
	cv_not_empty.notify_all();

		if (camera) {
			camera->stop();
			for (auto& kv : buffer_map) {
				for (size_t i = 0; i < kv.second.map_bases.size(); ++i) {
					munmap(kv.second.map_bases[i], kv.second.map_lengths[i]);
				}
			}
			camera->release();
			camera.reset();
		}
	cm.stop();

	close_socket(client);
	close_socket(server_fd);
	cleanup_sockets();
	return 0;
}
