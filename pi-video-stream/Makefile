CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS  = -lws2_32 -lstdc++fs   # <- needed for Windows (Winsock + filesystem)

# Targets
SERVER = server
CLIENT = client

# Source files
SERVER_SRC = server.cpp image_loader.cpp
CLIENT_SRC = client.cpp image_loader.cpp

# Object files
SERVER_OBJ = $(SERVER_SRC:.cpp=.o)
CLIENT_OBJ = $(CLIENT_SRC:.cpp=.o)

.PHONY: all clean run_server run_client

all: $(SERVER) $(CLIENT)

# Link binaries
$(SERVER): $(SERVER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CLIENT): $(CLIENT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run helpers (optional)
run_server: $(SERVER)
	./$(SERVER)

run_client: $(CLIENT)
	./$(CLIENT)

clean:
	rm -f $(SERVER_OBJ) $(CLIENT_OBJ) $(SERVER) $(CLIENT)
