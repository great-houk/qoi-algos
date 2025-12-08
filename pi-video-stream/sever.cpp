// C++ program to show the example of server application in
// socket programming
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>


int createConnection(sockaddr_in &server_addr, int &server_socket, int &client_socket) {
    int status;
    // creating socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);

    // specifying the address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    server_addr.sin_addr.s_addr = INADDR_ANY;


    // binding socket.
    status = bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if(!status) return status;

    // listening to the assigned socket
    status = listen(server_socket, 5);
    if(!status) return status;

    // accepting connection request and return client socket
    client_socket = accept(server_socket, nullptr, nullptr);

    //no error
    return 0;
}

int recieveMessage(std::vector<char>* buffer, int &client_socket) {
    // recieving data
    return recv(client_socket, buffer, buffer->size(), 0);

}


int closeConnection(int server_socket) {
    // closing the socket.    

    return close(server_socket);
}

int main() {
    int server_socket, client_socket, status;
    sockaddr_in server_addr;

    status = createConnection(server_addr, server_socket, client_socket);
    if(!status) return status;

    std::vector<char> buffer(10000);
    status = recieveMessage(&buffer, client_socket);

    

}




