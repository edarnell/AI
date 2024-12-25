# Compiler and Flags
CC = g++
CFLAGS = -Wall -Isrc -lbz2 -std=c++20

# Default Target
all:
	mkdir -p bin
	$(CC) $(CFLAGS) src/main.cpp src/utils.cpp src/Xi.cpp src/CPP.cpp src/N3R.cpp src/LM.cpp -o bin/xi

# Clean Up
clean:
	rm -rf bin
