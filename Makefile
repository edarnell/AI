# Compiler and Flags
CC = g++
CFLAGS = -std=c++17 -Wall -Isrc

# Default Target
all:
	mkdir -p bin
	$(CC) $(CFLAGS) src/Xi.cpp src/utils.cpp src/gpt.cpp src/CPP.cpp src/N3R.cpp src/LM.cpp -o bin/xi

# Clean Up
clean:
	rm -rf bin
