// C++ program to illustrate the client application in the
// socket programming
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <dirent.h>
#include <filesystem>
#include <fstream>

#include "image_loader.hpp"



struct ImageData {
	std::string path;
	std::vector<uint8_t> data;
	int size;
};


int createConnection(int &client_socket, sockaddr_in &server_addr) {
    int status;
    // creating socket
    client_socket = socket(AF_INET, SOCK_STREAM, 0);

    // specifying address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    // sending connection request
    status = connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr));

    return status;
}

std::vector<uint8_t>* grabNextPhotoData() {
    static int currPhoto = 0;
    initImages();

    return &images[currPhoto++].data;
}


int sendMessage(const char* message, int client_socket) {
    // sending data
    return send(client_socket, message, strlen(message), 0);

}

int main()
{
    int status;
    int client_socket;
    sockaddr_in server_addr;
    

    status = createConnection(client_socket, server_addr);
    if(!status) return status;

    
    status = sendMessage((char*) grabNextPhotoData(), client_socket);
    if(!status) return status;

    
    // closing socket
    close(client_socket);

    return 0;
}