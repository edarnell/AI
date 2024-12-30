#include <iostream>
#include <fstream>
#include <bzlib.h> // Library for bz2 decompression

void extractFirstRecords(const std::string& inputPath, const std::string& outputPath, size_t maxBytes) {
    FILE* file = fopen(inputPath.c_str(), "rb");
    if (!file) {
        std::cerr << "Error opening file: " << inputPath << "\n";
        return;
    }

    const int CHUNK_SIZE = 4096;
    char inputBuffer[CHUNK_SIZE];
    int bzError;
    BZFILE* bzf = BZ2_bzReadOpen(&bzError, file, 0, 0, nullptr, 0);

    if (bzError != BZ_OK) {
        std::cerr << "BZ2_bzReadOpen failed with code: " << bzError << "\n";
        fclose(file);
        return;
    }

    std::ofstream outFile(outputPath);
    if (!outFile) {
        std::cerr << "Error creating output file: " << outputPath << "\n";
        BZ2_bzReadClose(&bzError, bzf);
        fclose(file);
        return;
    }

    size_t totalBytesRead = 0;
    while (bzError == BZ_OK && totalBytesRead < maxBytes) {
        int bytesRead = BZ2_bzRead(&bzError, bzf, inputBuffer, CHUNK_SIZE);
        if (bzError == BZ_OK || bzError == BZ_STREAM_END) {
            size_t toWrite = std::min(static_cast<size_t>(bytesRead), maxBytes - totalBytesRead);
            outFile.write(inputBuffer, toWrite);
            totalBytesRead += toWrite;
        }
    }

    if (bzError != BZ_STREAM_END) {
        std::cerr << "Decompression ended with error code: " << bzError << "\n";
    } else {
        std::cout << "Decompression completed successfully.\n";
    }

    BZ2_bzReadClose(&bzError, bzf);
    fclose(file);
    outFile.close();
}

int main() {
    // Extract the first few megabytes for analysis
    extractFirstRecords("wiki.xml.bz2", "output_sample.xml", 5 * 1024 * 1024); // 5 MB
    std::cout << "Sample extraction completed. Check output_sample.xml for analysis.\n";
    return 0;
}
