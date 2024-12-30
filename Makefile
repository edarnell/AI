# Simplified Makefile for Xi project

all:
	g++ -Wall -Isrc -std=c++20 src/main.cpp src/utils.cpp src/zip.cpp src/Xi.cpp src/N3R.cpp src/LM.cpp -o bin/xi -lbz2 -lz
# gdb bin/xi
debug:
	g++ -Wall -Isrc -std=c++20 -g -fsanitize=address src/main.cpp src/utils.cpp src/zip.cpp src/Xi.cpp src/N3R.cpp src/LM.cpp -o bin/dbg -lbz2 -lz
# gdb bin/dbg
clean:
	rm -f bin/xi bin/dbg
