CXX ?= g++
PKG_CONFIG ?= pkg-config
MC_INCLUDE ?= ../multi-cpu
QOI_ROOT ?= ..
MC_SRC ?= $(MC_INCLUDE)/qoi-mc.cpp
OMPFLAGS ?= -fopenmp

CXXFLAGS ?= -std=c++20 -Wall -Wextra -O2 -I$(MC_INCLUDE) -I$(QOI_ROOT)
LDFLAGS ?=

SERVER_EXE ?= server.exe
CLIENT_EXE ?= client

# Resolve third-party flags via pkg-config.
SERVER_OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv4 2>/dev/null)
SERVER_OPENCV_LIBS   := $(shell $(PKG_CONFIG) --libs opencv4 2>/dev/null)
CLIENT_CAM_CFLAGS    := $(shell $(PKG_CONFIG) --cflags libcamera 2>/dev/null)
CLIENT_CAM_LIBS      := $(shell $(PKG_CONFIG) --libs libcamera 2>/dev/null)

.PHONY: all server client clean

all: server

server: $(SERVER_EXE)

$(SERVER_EXE): server.cpp $(MC_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(SERVER_OPENCV_CFLAGS) $^ -o $@ $(LDFLAGS) $(OMPFLAGS) $(SERVER_OPENCV_LIBS) -lws2_32

client: $(CLIENT_EXE)

$(CLIENT_EXE): client.cpp $(MC_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(CLIENT_CAM_CFLAGS) $^ -o $@ $(LDFLAGS) $(OMPFLAGS) $(CLIENT_CAM_LIBS) -pthread

clean:
	rm -f $(SERVER_EXE) $(CLIENT_EXE)
