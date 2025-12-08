// ---- Platform-specific networking includes ----
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <unistd.h>
#endif

// ---- Standard includes ----
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

#define PORT 8080

// ---- Cross-platform helpers ----
#ifdef _WIN32
using socket_t = SOCKET;

static bool init_sockets() {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2,2), &wsa) == 0;
}

static void cleanup_sockets() {
    WSACleanup();
}

static void close_socket(socket_t s) {
    closesocket(s);
}

static int64_t recv_wrapper(socket_t s, void* buf, size_t len) {
    return recv(s, static_cast<char*>(buf), (int)len, 0);
}

#else  // LINUX/UNIX
using socket_t = int;

static bool init_sockets() { return true; }
static void cleanup_sockets() {}

static void close_socket(socket_t s) {
    close(s);
}

static int64_t recv_wrapper(socket_t s, void* buf, size_t len) {
    return read(s, buf, len);
}

#endif


int main() {
    if (!init_sockets()) {
        std::cerr << "Failed to initialize sockets\n";
        return -1;
    }

    socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Socket creation failed\n";
        cleanup_sockets();
        return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Bind failed\n";
        close_socket(server_fd);
        cleanup_sockets();
        return -1;
    }

    listen(server_fd, 1);
    std::cout << "Listening...\n";

    socket_t client = accept(server_fd, nullptr, nullptr);
    if (client < 0) {
        std::cerr << "Accept failed\n";
        close_socket(server_fd);
        cleanup_sockets();
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    size_t total = 0;
    std::vector<uint8_t> buf(4096);

    while (true) {
        int64_t n = recv_wrapper(client, buf.data(), buf.size());
        if (n <= 0)
            break;
        total += n;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Received " << total << " bytes in " << seconds << " seconds\n";

    close_socket(client);
    close_socket(server_fd);
    cleanup_sockets();
}
