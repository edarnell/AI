#include "utils.h"
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <bzlib.h>

namespace {
    // SHA-256 Constants
    const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // Rotate right operation
    inline uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    // Padding for the message
    std::vector<uint8_t> pad(const std::string &input) {
        std::vector<uint8_t> padded(input.begin(), input.end());
        padded.push_back(0x80); // Append 1 bit followed by 7 zero bits
        while ((padded.size() * 8) % 512 != 448) {
            padded.push_back(0x00); // Pad with zeros
        }

        uint64_t bitLength = input.size() * 8;
        for (int i = 7; i >= 0; --i) {
            padded.push_back(static_cast<uint8_t>(bitLength >> (i * 8)));
        }

        return padded;
    }

} // namespace

std::string sha256(const std::string &input) {
    // Initial hash values
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    std::vector<uint8_t> padded = pad(input);

    // Process each 512-bit chunk
    for (size_t i = 0; i < padded.size(); i += 64) {
        uint32_t w[64] = {0};
        for (size_t j = 0; j < 16; ++j) {
            w[j] = (padded[i + j * 4 + 0] << 24) |
                   (padded[i + j * 4 + 1] << 16) |
                   (padded[i + j * 4 + 2] << 8) |
                   (padded[i + j * 4 + 3] << 0);
        }

        for (size_t j = 16; j < 64; ++j) {
            uint32_t s0 = rotr(w[j - 15], 7) ^ rotr(w[j - 15], 18) ^ (w[j - 15] >> 3);
            uint32_t s1 = rotr(w[j - 2], 17) ^ rotr(w[j - 2], 19) ^ (w[j - 2] >> 10);
            w[j] = w[j - 16] + s0 + w[j - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h0 = h[7];

        for (size_t j = 0; j < 64; ++j) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h0 + S1 + ch + k[j] + w[j];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h0 = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h0;
    }

    std::stringstream hashStream;
    for (uint32_t value : h) {
        hashStream << std::hex << std::setw(8) << std::setfill('0') << value;
    }
    return hashStream.str();
}
namespace Utils {
        std::vector<std::pair<int64_t, std::string>> readTopic(const std::string& filePath, const std::string& topic) {
        BZFILE* file = BZ2_bzopen(filePath.c_str(), "rb");
        if (!file) throw std::runtime_error("Unable to open file for reading.");

        constexpr int BUFFER_SIZE = 4096;
        char buffer[BUFFER_SIZE];
        std::string decompressedData;
        int bytesRead;

        while ((bytesRead = BZ2_bzread(file, buffer, BUFFER_SIZE)) > 0) {
            decompressedData.append(buffer, bytesRead);

            // Locate the topic marker
            size_t topicPos = decompressedData.find("Topic: " + topic);
            if (topicPos != std::string::npos) {
                size_t nextTopicPos = decompressedData.find("Topic: ", topicPos + 1);
                std::string topicData = decompressedData.substr(topicPos, nextTopicPos - topicPos);
                BZ2_bzclose(file);

                // Parse the topic data into timestamp/message pairs
                std::vector<std::pair<int64_t, std::string>> messages;
                std::istringstream topicStream(topicData);
                std::string line;

                while (std::getline(topicStream, line)) {
                    if (line.starts_with("Topic: ")) continue; // Skip header
                    size_t delimPos = line.find('|');
                    if (delimPos != std::string::npos) {
                        int64_t timestamp = std::stoll(line.substr(0, delimPos));
                        std::string message = line.substr(delimPos + 1);
                        messages.emplace_back(timestamp, message);
                    }
                }
                return messages;
            }
        }

        BZ2_bzclose(file);
        throw std::runtime_error("Topic not found.");
    }
    
    std::string extractField(const std::string& input, const std::string& fieldName) {
        size_t fieldStart = input.find("\"" + fieldName + "\":");
        if (fieldStart == std::string::npos) return "";

        size_t valueStart = input.find("\"", fieldStart + fieldName.length() + 3);
        size_t valueEnd = input.find("\"", valueStart + 1);

        if (valueStart == std::string::npos || valueEnd == std::string::npos) return "";
        return input.substr(valueStart + 1, valueEnd - valueStart - 1);
    }

    std::unordered_map<std::string, std::vector<std::string>> chat(
    const std::string& filePath,
    std::unordered_map<std::string, std::string>& nodeData) 
    {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open chat data file.");
        }

        std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        std::unordered_map<std::string, std::vector<std::string>> parentChildMap;

        size_t mappingStart = jsonStr.find("\"mapping\":");
        if (mappingStart != std::string::npos) {
            size_t mappingEnd = jsonStr.find("]", mappingStart);
            std::string mappingContent = jsonStr.substr(mappingStart, mappingEnd - mappingStart + 1);

            size_t nodeStart = 0;
            while ((nodeStart = mappingContent.find("\"id\":", nodeStart)) != std::string::npos) {
                size_t nodeEnd = mappingContent.find("}", nodeStart);
                std::string nodeContent = mappingContent.substr(nodeStart, nodeEnd - nodeStart + 1);

                std::string id = Utils::extractField(nodeContent, "id");
                std::string parent = Utils::extractField(nodeContent, "parent");
                std::string role = Utils::extractField(nodeContent, "role");
                std::string content = Utils::extractField(nodeContent, "content");

                if (!id.empty()) {
                    nodeData[id] = content;
                    if (!parent.empty()) {
                        parentChildMap[parent].push_back(id);
                    }
                }

                nodeStart = nodeEnd + 1;
            }
        }

        return parentChildMap;
    }

    
    void appendToBzip2(const std::string& filePath, const std::string& topic, const std::vector<std::pair<int64_t, std::string>>& messages) {
        BZFILE* file = BZ2_bzopen(filePath.c_str(), "ab");
        if (!file) throw std::runtime_error("Unable to open file for appending.");

        std::ostringstream oss;
        oss << "Topic: " << topic << "\n";
        for (const auto& [timestamp, message] : messages) {
            oss << timestamp << "|" << message << "\n";
        }

        std::string topicData = oss.str();
        BZ2_bzwrite(file, topicData.data(), topicData.size());
        BZ2_bzclose(file);
    }
    
    Log logL = Log::ERROR; // Default logging level

    void log(Log l, const std::string& m) {
        if (l <= logL) {
            std::ofstream f("log.txt", std::ios::out | std::ios::app); // Append mode
            if (!f) {
                std::cerr << "Failed to open log file for writing." << std::endl;
                return; // Avoid throwing during logging
            }
            f << m << std::endl;
            f.close();

            // Optional: Print to console for higher log levels
            if (l == Log::ERROR || l == Log::INFO) {
                std::cout << m << std::endl;
            }
        }
    }

    void setLog(Log l) {
        logL = l;
    }
}

