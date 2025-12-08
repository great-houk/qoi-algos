CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -lstdc++fs

# Targets
SERVER = server
CLIENT = client

# Source files
SERVER_SRC = server.cpp image_loader.cpp
CLIENT_SRC = client.cpp image_loader.cpp

# Object files
SERVER_OBJ = $(SERVER_SRC:.cpp=.o)
CLIENT_OBJ = $(CLIENT_SRC:.cpp=.o)

.PHONY: all server client clean

all: $(SERVER) $(CLIENT)

server: $(SERVER)
	./$(SERVER)

client: $(CLIENT)
	./$(CLIENT)

$(SERVER): $(SERVER_OBJ)
	$(CXX) $(CXXFLAGS) -o $(SERVER) $(SERVER_OBJ) $(LDFLAGS)

$(CLIENT): $(CLIENT_OBJ)
	$(CXX) $(CXXFLAGS) -o $(CLIENT) $(CLIENT_OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(SERVER_OBJ) $(CLIENT_OBJ) $(SERVER) $(CLIENT)

.PHONY: all server client clean
