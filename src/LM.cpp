#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "LM.h"
#include "utils.h"

// Constructor LM(int dim, float lr, float reg, float noise);  LM model(50, 0.01, 0.001, 0.01); - in Xi.cpp
LM::LM(int d,float l,float r,float n)
{
    dim=d;
    lr=l;
    reg=r;
    noise=n;
}
    
// Generate a random float in the range [0, 1]
float LM::randomFloat() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0, 1.0);
    return dis(gen);
}

// Normalize a vector to unit length
void LM::normalizeVector(std::vector<float>& vec) {
    float magnitude = std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0f, [](float sum, float val) {
        return sum + val * val;
    }));
    if (magnitude > 0) {
        for (auto& val : vec) {
            val /= magnitude;
        }
    }
}

// Add noise to a vector
void LM::addNoise(std::vector<float>& vec, float factor) {
    for (auto& val : vec) {
        val += factor * randomFloat();
    }
}

// Add a word with random initialization
void LM::addWord(const std::string& word) {
    if (embeddings.find(word) == embeddings.end()) {
        embeddings[word] = std::vector<float>(dim);
        for (auto& val : embeddings[word]) {
            val = randomFloat();
        }
        normalizeVector(embeddings[word]);
    }
}

// Retrieve the embedding vector for a given word
const std::vector<float>& LM::getEmbedding(const std::string& word) const {
    if (embeddings.find(word) == embeddings.end()) {
        throw std::runtime_error("Word not found in embeddings.");
    }
    return embeddings.at(word);
}

// Update embeddings using a context-aware competitive learning algorithm
void LM::updateWithContext(const std::vector<std::string>& contexts, const std::string& word, const std::string& contextWord, float coOccurrence) {
    if (embeddings.find(word) == embeddings.end() || embeddings.find(contextWord) == embeddings.end()) {
        return; // Skip updates for unknown words
    }

    auto& wordVec = embeddings[word];
    auto& contextVec = embeddings[contextWord];
    std::vector<float> pooledContext = getContextEmbedding(contexts);

    for (int i = 0; i < dim; ++i) {
        float gradient = (pooledContext[i] * wordVec[i] * contextVec[i]) - reg;
        wordVec[i] += lr * gradient;
        contextVec[i] += lr * gradient;
    }

    addNoise(wordVec, noise);
    addNoise(contextVec, noise);
    normalizeVector(wordVec);
    normalizeVector(contextVec);
}

// Pool embeddings for a set of contexts
std::vector<float> LM::getContextEmbedding(const std::vector<std::string>& contexts) const {
    std::vector<float> pooled(dim, 0.0f);
    for (const auto& ctx : contexts) {
        if (embeddings.find(ctx) != embeddings.end()) {
            for (int i = 0; i < dim; ++i) {
                pooled[i] += embeddings.at(ctx)[i];
            }
        }
    }
    for (float& val : pooled) {
        val /= contexts.size();
    }
    return pooled;
}

// Competitive update for embeddings
void LM::competitiveUpdate() {
    for (auto& [word, vec] : embeddings) {
        float maxVal = *std::max_element(vec.begin(), vec.end());
        for (float& val : vec) {
            val = (val == maxVal) ? val + lr : val * (1.0f - reg);
        }
        normalizeVector(vec);
    }
}

// Train embeddings for a given co-occurrence dataset
void LM::train(const CoOccurrenceData& data, size_t epochs) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& [word, contextWord, coOccurrence] : data) {
            updateWithContext({contextWord}, word, contextWord, coOccurrence);
        }
        competitiveUpdate();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " complete." << std::endl;
    }
}

// Save embeddings to a JSON file
void LM::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving embeddings.");
    }

    file << "{\n";
    bool firstWord = true;
    for (const auto& [word, vec] : embeddings) {
        if (!firstWord) {
            file << ",\n";
        }
        firstWord = false;

        file << "  \"" << word << "\": [";
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) {
                file << ", ";
            }
        }
        file << "]";
    }
    file << "\n}\n";
}

// Load embeddings from a JSON file
void LM::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading embeddings.");
    }

    embeddings.clear();
    std::string line, word;
    while (std::getline(file, line)) {
        size_t keyStart = line.find("\"");
        if (keyStart == std::string::npos) continue;
        size_t keyEnd = line.find("\"", keyStart + 1);
        word = line.substr(keyStart + 1, keyEnd - keyStart - 1);

        size_t vecStart = line.find("[");
        size_t vecEnd = line.find("]");
        if (vecStart == std::string::npos || vecEnd == std::string::npos) continue;

        std::string vecContent = line.substr(vecStart + 1, vecEnd - vecStart - 1);
        std::istringstream stream(vecContent);
        std::vector<float> vec;
        float value;
        while (stream >> value) {
            vec.push_back(value);
            if (stream.peek() == ',') {
                stream.ignore();
            }
        }
        embeddings[word] = vec;
    }
}

// Serialize embeddings to a binary string
std::string LM::serialize() const {
    std::ostringstream oss(std::ios::binary);
    int version = 1;
    oss.write(reinterpret_cast<const char*>(&version), sizeof(version));

    size_t mapSize = embeddings.size();
    oss.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (const auto& [word, vec] : embeddings) {
        size_t wordLen = word.size();
        oss.write(reinterpret_cast<const char*>(&wordLen), sizeof(wordLen));
        oss.write(word.data(), wordLen);

        size_t vecSize = vec.size();
        oss.write(reinterpret_cast<const char*>(&vecSize), sizeof(vecSize));
        oss.write(reinterpret_cast<const char*>(vec.data()), vecSize * sizeof(float));
    }

    return oss.str();
}

// Deserialize embeddings from a binary string
void LM::deserialize(const std::string& data) {
    std::istringstream iss(data, std::ios::binary);
    int version;
    iss.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported model version.");
    }

    size_t mapSize;
    iss.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

    embeddings.clear();
    for (size_t i = 0; i < mapSize; ++i) {
        size_t wordLen;
        iss.read(reinterpret_cast<char*>(&wordLen), sizeof(wordLen));
        std::string word(wordLen, '\0');
        iss.read(&word[0], wordLen);

        size_t vecSize;
        iss.read(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));
        std::vector<float> vec(vecSize);
        iss.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(float));

        embeddings[word] = vec;
    }

    if (iss.fail()) {
        throw std::runtime_error("Failed to deserialize model.");
    }
}






