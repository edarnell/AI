#include "zip.h"
#include <stdexcept>
#include <iostream>
#include <zlib.h>
#include <fstream>
#include <iterator> // Include for std::istream_iterator
#include <string>   // Include for std::string

Zip::Zip(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open zip file: " + filePath);
    }

    file.unsetf(std::ios::skipws); // Disable skipping whitespace
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()); // Use streambuf iterator
}

// File Header Structure
struct FH {
    uint32_t sig;
    uint16_t vNeeded;
    uint16_t gpFlag;
    uint16_t method;
    uint16_t modTime;
    uint16_t modDate;
    uint32_t crc32;
    uint32_t cSize;
    uint32_t uSize;
    uint16_t nLen;
    uint16_t xLen;
};

void Zip::ext(const std::string& fn, const std::function<void(const char*, int)>& cb) {
    int i = 0; // Signed input index or offset
    if (i < 0) {
        throw std::runtime_error("Index 'i' is negative, which is invalid.");
    }

    uint64_t l = static_cast<uint64_t>(i); // Cast to unsigned after validation

    while (l < data.size()) {
        if (l + 30 > data.size()) {
            throw std::runtime_error("Unexpected end of data while reading file header");
        }

        // Process using 'l'
        uint16_t nameLen = data[l + 26] | (data[l + 27] << 8);
        uint16_t extraLen = data[l + 28] | (data[l + 29] << 8);
        uint32_t compressedSize = data[l + 18] | (data[l + 19] << 8) | (data[l + 20] << 16) | (data[l + 21] << 24);

        if (l + 30 + nameLen > data.size()) {
            throw std::runtime_error("Unexpected end of data while reading file name");
        }

        std::string currentName(reinterpret_cast<const char*>(&data[l + 30]), nameLen);

        if (currentName == fn) {
            size_t offset = l + 30 + nameLen + extraLen;
            if (offset + compressedSize > data.size()) {
                throw std::runtime_error("Unexpected end of data while reading file content");
            }

            cb(reinterpret_cast<const char*>(&data[offset]), compressedSize);
            return;
        }

        l += 30 + nameLen + extraLen + compressedSize;
    }

    throw std::runtime_error("File not found in archive: " + fn);
}


void Zip::procData(int i, int cS, int uS, const std::function<void(const char*, int)>& p) {
    if (i < 0 || cS < 0 || uS < 0) {
        throw std::runtime_error("Negative value in index, compressed size, or uncompressed size");
    }

    uint64_t l = static_cast<uint64_t>(i); // Cast to unsigned after validation

    if (l + sizeof(FH) > data.size()) {
        throw std::runtime_error("Offset exceeds data size at header parsing");
    }

    FH* h = reinterpret_cast<FH*>(&data[l]);
    std::cout << "Processing local file header at offset: " << i << std::endl;
    std::cout << "nLen: " << h->nLen << ", xLen: " << h->xLen << std::endl;

    l += 30 + h->nLen + h->xLen;
    if (l + cS > data.size()) {
        throw std::runtime_error("Compressed data offset exceeds file size");
    }

    z_stream z{};
    z.next_in = reinterpret_cast<Bytef*>(&data[l]); // Use 'l' as the offset
    z.avail_in = cS;

    if (inflateInit2(&z, -MAX_WBITS) != Z_OK) throw std::runtime_error("zlib init failed");

    const int bS = 4096; // Buffer size
    char b[bS];

    do {
        z.next_out = reinterpret_cast<Bytef*>(b);
        z.avail_out = bS;
        int r = inflate(&z, Z_NO_FLUSH);
        if (r == Z_STREAM_ERROR || r == Z_DATA_ERROR || r == Z_MEM_ERROR) {
            inflateEnd(&z);
            throw std::runtime_error("Decompression error");
        }
        p(b, bS - z.avail_out);
    } while (z.avail_out == 0);

    inflateEnd(&z);
}

void Zip::read(const std::vector<unsigned char>& d) {
    data = d;
    size_t o = data.size() - 22; // Central directory offset
    if (o > data.size()) throw std::runtime_error("Invalid central directory offset");

    uint32_t cO = *reinterpret_cast<uint32_t*>(&data[o + 16]); // Central directory start
    std::cout << "Central directory offset (raw): " << cO << std::endl;

    o = cO;
    while (o < data.size()) {
        if (o + 46 > data.size()) throw std::runtime_error("Central directory record exceeds file size");
        uint16_t nLen = *reinterpret_cast<uint16_t*>(&data[o + 28]);
        uint16_t xLen = *reinterpret_cast<uint16_t*>(&data[o + 30]);
        uint16_t cLen = *reinterpret_cast<uint16_t*>(&data[o + 32]);

        uint32_t cS = *reinterpret_cast<uint32_t*>(&data[o + 20]);
        uint32_t uS = *reinterpret_cast<uint32_t*>(&data[o + 24]);
        uint32_t lO = *reinterpret_cast<uint32_t*>(&data[o + 42]);

        procData(lO, cS, uS, [](const char* buf, int sz) {
            std::cout.write(buf, sz);
        });

        o += 46 + nLen + xLen + cLen;
    }
}


