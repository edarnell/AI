#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <random>
#include <stdexcept>
#include <numeric>
#include <fstream>
#include <iomanip>

// Full SHA-256 hash implementation
    std::string sha256(const std::string &input);
// Helper function to trim whitespace
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n");
    size_t last = str.find_last_not_of(" \t\n");
    return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, last - first + 1);
}

// Generate a random float
inline float randomFloat(float min = -0.05f, float max = 0.05f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

// Normalize a vector to unit length
inline void normalizeVector(std::vector<float>& vec) {
    float magnitude = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
    if (magnitude > 0.0f) {
        for (float& v : vec) {
            v /= magnitude;
        }
    }
}

// Add stochastic noise to a vector
inline void addNoise(std::vector<float>& vec, float factor, float min = -1.0f, float max = 1.0f) {
    for (float& v : vec) {
        v += randomFloat(min, max) * factor;
    }
}

// Generic function to find the matching closing bracket in a tokenized list
template <typename Container, typename Accessor>
inline size_t FindClose(const Container& tokens, size_t start, char open, char close, Accessor accessor) {
    int depth = 0;
    for (size_t i = start; i < tokens.size(); ++i) {
        char current = accessor(tokens[i]);
        if (current == open) depth++;
        if (current == close) depth--;
        if (depth == 0) return i;
    }
    throw std::runtime_error("Mismatched brackets detected.");
}

namespace Utils {
    std::unordered_map<std::string, std::vector<std::string>> chat(const std::string& filePath, std::unordered_map<std::string, std::string>& nodeData);
    std::vector<std::pair<int64_t, std::string>> readTopic(const std::string& filePath, const std::string& topic);
    void appendToBzip2(const std::string& filePath, const std::string& topic, const std::vector<std::pair<int64_t, std::string>>& messages);
}


#endif // UTILS_H


