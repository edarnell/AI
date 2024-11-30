# Compiler and Flags
CC = g++
CFLAGS = -std=c++17 -Icore -Iparsers

# Default Target
all:
	mkdir -p bin
	$(CC) $(CFLAGS) parsers/gpt.cpp core/N3R.cpp core/S3R.cpp core/L3R.cpp -o bin/gpt

# Clean Up
clean:
	rm -rf bin
