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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "image_loader.hpp"

#define PORT 8080

// ---- Small cross-platform helpers ----
#ifdef _WIN32
using socket_t = SOCKET;
static bool init_sockets()
{
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}
static void cleanup_sockets()
{
    WSACleanup();
}
static void close_socket(socket_t s)
{
    closesocket(s);
}
#else
using socket_t = int;
static bool init_sockets()
{
    return true; // nothing to do on POSIX
}
static void cleanup_sockets()
{
    // nothing to do on POSIX
}
static void close_socket(socket_t s)
{
    close(s);
}
#endif

std::vector<uint8_t>* grabNextPhotoData()
{
    static int currPhoto = 0;
    initImages();

    auto ret = &images[currPhoto++].data;
    currPhoto %= images.size();
    return ret;
}

int main()
{
    if (!init_sockets()) {
        std::cerr << "Failed to initialize socket subsystem" << std::endl;
        return -1;
    }

    socket_t sock;
    struct sockaddr_in serv_addr {};

    // Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);

#ifdef _WIN32
    if (sock == INVALID_SOCKET) {
        std::cerr << "Socket creation error" << std::endl;
        cleanup_sockets();
        return -1;
    }
#else
    if (sock < 0) {
        std::cerr << "Socket creation error" << std::endl;
        cleanup_sockets();
        return -1;
    }
#endif

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 address from text to binary form
    if (inet_pton(AF_INET, "192.168.137.89", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address / Address not supported" << std::endl;
        close_socket(sock);
        cleanup_sockets();
        return -1;
    }

    // Connect to server
    if (connect(sock, reinterpret_cast<struct sockaddr*>(&serv_addr),
                sizeof(serv_addr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        close_socket(sock);
        cleanup_sockets();
        return -1;
    }

    // Send data
    for (int i = 0; i < 50; i++) {
        std::vector<uint8_t>* photoData = grabNextPhotoData();
        send(sock,
             reinterpret_cast<const char*>(photoData->data()),
             static_cast<int>(photoData->size()),
             0);
        std::cout << "Message sent" << std::endl;
    }

    // Close socket
    close_socket(sock);
    cleanup_sockets();
    return 0;
}
