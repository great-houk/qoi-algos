CXX      = g++
OS_NAME := $(shell uname -s 2>/dev/null || echo Windows_NT)

# --- OpenCV discovery (pkg-config only) ---
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)

LIBCAMERA_CFLAGS := $(shell pkg-config --cflags libcamera 2>/dev/null)
LIBCAMERA_LIBS   := $(shell pkg-config --libs libcamera 2>/dev/null)
ifeq ($(strip $(LIBCAMERA_CFLAGS)),)
	LIBCAMERA_CFLAGS := -I/usr/include/libcamera
endif
ifeq ($(strip $(LIBCAMERA_LIBS)),)
	LIBCAMERA_LIBS := -lcamera -lcamera-base
endif

# --- Platform-specific flags ---
WSOCK_LIB =
PTHREAD_FLAG =
ifneq (,$(findstring NT,$(OS_NAME)))
	WSOCK_LIB = -lws2_32
else
	PTHREAD_FLAG = -pthread
endif

CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fopenmp $(PTHREAD_FLAG) -I.. -I../multi-cpu

SERVER_CXXFLAGS = $(CXXFLAGS) $(OPENCV_CFLAGS)
CLIENT_CXXFLAGS = $(CXXFLAGS) $(LIBCAMERA_CFLAGS)

SERVER_LDFLAGS  = $(WSOCK_LIB) $(OPENCV_LIBS) -fopenmp $(PTHREAD_FLAG)
CLIENT_LDFLAGS  = $(WSOCK_LIB) $(LIBCAMERA_LIBS) -lcamera -lcamera-base -fopenmp $(PTHREAD_FLAG)

# Targets
SERVER = server
CLIENT = client

COMMON_OBJ = qoi-mc.o

# Source files
SERVER_SRC = server.cpp
CLIENT_SRC = client.cpp

# Object files
SERVER_OBJ = $(SERVER_SRC:.cpp=.o) $(COMMON_OBJ)
CLIENT_OBJ = $(CLIENT_SRC:.cpp=.o) $(COMMON_OBJ)

.PHONY: all clean run_server run_client

all: $(SERVER) $(CLIENT)

# Link binaries
$(SERVER): $(SERVER_OBJ)
	$(CXX) $(SERVER_CXXFLAGS) -o $@ $^ $(SERVER_LDFLAGS)

$(CLIENT): $(CLIENT_OBJ)
	$(CXX) $(CLIENT_CXXFLAGS) -o $@ $^ $(CLIENT_LDFLAGS)

# Compile objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(COMMON_OBJ): ../multi-cpu/qoi-mc.cpp ../multi-cpu/qoi-mc.hpp ../qoi.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run helpers (optional)
run_server: $(SERVER)
	./$(SERVER)

run_client: $(CLIENT)
	./$(CLIENT)

clean:
	rm -f $(SERVER_OBJ) $(CLIENT_OBJ) $(SERVER) $(CLIENT)
