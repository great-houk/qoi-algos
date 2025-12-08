// Save as server.cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <chrono>

#define PORT 8080

int main() {
	int server_fd = socket(AF_INET, SOCK_STREAM, 0);
	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(PORT);

	bind(server_fd, (sockaddr*)&addr, sizeof(addr));
	listen(server_fd, 1);

	std::cout << "Listening...\n";

	int client = accept(server_fd, nullptr, nullptr);

	auto start = std::chrono::high_resolution_clock::now();

	size_t total = 0;
	std::vector<uint8_t> buf(4096);

	while (true) {
		ssize_t n = read(client, buf.data(), buf.size());
		if (n <= 0)
			break;
		total += n;
	}

	auto end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(end - start).count();

	std::cout << "Received " << total << " bytes in " << seconds
			  << " seconds\n";

	close(client);
	close(server_fd);
}
