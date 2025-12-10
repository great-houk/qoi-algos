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
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../multi-cpu/qoi-mc.hpp"

#define PORT 8080

// ---- Cross-platform helpers ----
#ifdef _WIN32
using socket_t = SOCKET;

static bool init_sockets() {
	WSADATA wsa;
	return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}

static void cleanup_sockets() { WSACleanup(); }

static void close_socket(socket_t s) { closesocket(s); }

static int64_t recv_wrapper(socket_t s, void* buf, size_t len) {
	return recv(s, static_cast<char*>(buf), static_cast<int>(len), 0);
}

#else  // LINUX/UNIX
using socket_t = int;

static bool init_sockets() { return true; }
static void cleanup_sockets() {}

static void close_socket(socket_t s) { close(s); }

static int64_t recv_wrapper(socket_t s, void* buf, size_t len) {
	return read(s, buf, len);
}

#endif

static bool recv_all(socket_t s, uint8_t* buf, size_t len) {
	size_t total = 0;
	while (total < len) {
		int64_t n = recv_wrapper(s, buf + total, len - total);
		if (n <= 0) return false;
		total += static_cast<size_t>(n);
	}
	return true;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <pi_ip>\n";
		return -1;
	}
	std::string pi_ip = argv[1];

	if (!init_sockets()) {
		std::cerr << "Failed to initialize sockets\n";
		return -1;
	}

	socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0) {
		std::cerr << "Socket creation failed\n";
		cleanup_sockets();
		return -1;
	}

	sockaddr_in addr {};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(PORT);
	if (inet_pton(AF_INET, pi_ip.c_str(), &addr.sin_addr) <= 0) {
		std::cerr << "Invalid IP address: " << pi_ip << "\n";
		close_socket(sock);
		cleanup_sockets();
		return -1;
	}

	std::cout << "Connecting to Pi at " << pi_ip << ":" << PORT << "...\n";
	if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
		std::cerr << "Connection failed\n";
		close_socket(sock);
		cleanup_sockets();
		return -1;
	}

	MultiCPUQOI decoder;
	std::cout << "Connected. Receiving frames..." << std::endl;

	while (true) {
		uint32_t net_size = 0;
		if (!recv_all(sock, reinterpret_cast<uint8_t*>(&net_size),
					  sizeof(net_size))) {
			break;
		}
		uint32_t payload_size = ntohl(net_size);
		if (payload_size == 0) break;

		std::vector<uint8_t> encoded(payload_size);
		if (!recv_all(sock, encoded.data(), encoded.size())) {
			break;
		}

		QOIDecoderSpec spec {};
		std::vector<uint8_t> decoded = decoder.decode(encoded, spec);
		if (decoded.empty()) {
			std::cerr << "Failed to decode frame" << std::endl;
			break;
		}

		int type = (spec.channels == 4) ? CV_8UC4 : CV_8UC3;
		cv::Mat rgb(spec.height, spec.width, type, decoded.data());
		cv::Mat bgr;
		if (spec.channels == 4) {
			cv::cvtColor(rgb, bgr, cv::COLOR_RGBA2BGRA);
		} else {
			cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
		}

		cv::imshow("QOI Stream", bgr);
		if (cv::waitKey(1) == 27) {  // ESC to exit
			break;
		}
	}

	close_socket(sock);
	cleanup_sockets();
	return 0;
}
