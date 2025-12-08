#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#define PORT 8080

int main() {
	int server_fd, new_socket;
	struct sockaddr_in address;
	int opt = 1;
	int addrlen = sizeof(address);
	char buffer[1024] = {0};

	// Create socket
	server_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (server_fd == 0) {
		perror("socket failed");
		exit(EXIT_FAILURE);
	}

	// Attach socket to the port
	setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(PORT);

	// Bind
	if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
		perror("bind failed");
		exit(EXIT_FAILURE);
	}

	// Listen
	if (listen(server_fd, 3) < 0) {
		perror("listen");
		exit(EXIT_FAILURE);
	}

	// Accept a connection
	new_socket =
		accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
	if (new_socket < 0) {
		perror("accept");
		exit(EXIT_FAILURE);
	}

	// Read data
	read(new_socket, buffer, 1024);
	std::cout << "Message from client: " << buffer << std::endl;

	// Close socket
	close(new_socket);
	close(server_fd);
	return 0;
}