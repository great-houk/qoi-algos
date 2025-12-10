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
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../multi-cpu/qoi-mc.hpp"

#define PORT 8080

// ---- Small cross-platform helpers ----
#ifdef _WIN32
using socket_t = SOCKET;
static bool init_sockets() {
	WSADATA wsa;
	return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}
static void cleanup_sockets() { WSACleanup(); }
static void close_socket(socket_t s) { closesocket(s); }
#else
using socket_t = int;
static bool init_sockets() { return true; }
static void cleanup_sockets() {}
static void close_socket(socket_t s) { close(s); }
#endif

static bool send_all(socket_t s, const uint8_t* data, size_t len) {
	size_t total_sent = 0;
	while (total_sent < len) {
#ifdef _WIN32
		int sent = send(s, reinterpret_cast<const char*>(data + total_sent),
						static_cast<int>(len - total_sent), 0);
		if (sent <= 0) return false;
#else
		ssize_t sent =
			send(s, reinterpret_cast<const char*>(data + total_sent),
				 len - total_sent, 0);
		if (sent <= 0) return false;
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

	sockaddr_in addr {};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(PORT);

	if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
		0) {
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

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Failed to open webcam" << std::endl;
		close_socket(client);
		close_socket(server_fd);
		cleanup_sockets();
		return -1;
	}

	struct Frame {
		std::vector<uint8_t> pixels;  // RGB bytes
		uint32_t width;
		uint32_t height;
	};

	std::deque<Frame> queue;
	std::mutex mtx;
	std::condition_variable cv_not_full;
	std::condition_variable cv_not_empty;
	const size_t kMaxQueue = 3;  // keep a small buffer to avoid lag
	std::atomic<bool> running{true};

	// Capture thread: grabs frames, converts to RGB, and enqueues raw pixels.
	std::thread capture_thread([&]() {
		while (running) {
			cv::Mat frame;
			if (!cap.read(frame)) {
				std::cerr << "Failed to read frame from webcam" << std::endl;
				running = false;
				cv_not_empty.notify_all();
				break;
			}

			cv::Mat rgb;
			cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

			Frame f;
			f.width = static_cast<uint32_t>(rgb.cols);
			f.height = static_cast<uint32_t>(rgb.rows);
			size_t bytes = rgb.total() * rgb.elemSize();
			f.pixels.resize(bytes);
			if (rgb.isContinuous()) {
				std::memcpy(f.pixels.data(), rgb.data, bytes);
			} else {
				size_t row_bytes = static_cast<size_t>(rgb.cols * rgb.elemSize());
				for (int r = 0; r < rgb.rows; ++r) {
					std::memcpy(f.pixels.data() + r * row_bytes,
								rgb.ptr(r),
								row_bytes);
				}
			}

			std::unique_lock<std::mutex> lock(mtx);
			cv_not_full.wait(lock, [&] { return !running || queue.size() < kMaxQueue; });
			if (!running) break;
			queue.push_back(std::move(f));
			cv_not_empty.notify_one();
		}
	});

	// Encode/send thread: pops frames, encodes with multi-core QOI, sends.
	MultiCPUQOI encoder;
	std::thread encode_thread([&]() {
		while (running) {
			Frame f;
			{
				std::unique_lock<std::mutex> lock(mtx);
				cv_not_empty.wait(lock, [&] { return !running || !queue.empty(); });
				if (!running && queue.empty()) break;
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

			std::vector<uint8_t> encoded = encoder.encode(f.pixels, spec);
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
		}
	});

	std::cout << "Streaming frames..." << std::endl;

	encode_thread.join();
	running = false;
	cv_not_full.notify_all();
	cv_not_empty.notify_all();
	capture_thread.join();

	close_socket(client);
	close_socket(server_fd);
	cleanup_sockets();
	return 0;
}
