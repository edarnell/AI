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
#include "utils.h"

namespace LM {

    class Model {
    private:
        std::unordered_map<std::string, std::vector<float>> embeddings; // Word embeddings
        size_t dimension; // Dimensionality of embeddings
        float learningRate; // Learning rate for updates
        float regularization; // Regularization factor for training
        float noiseFactor; // Noise factor for stochastic update
    public:
        // Constructor: Initialize the model
        Model(size_t dim = 50, float lr = 0.01, float reg = 0.001, float noise = 0.01)
            : dimension(dim), learningRate(lr), regularization(reg), noiseFactor(noise) {}
    // Add a word with random initialization
    void addWord(const std::string& word) {
        if (embeddings.find(word) == embeddings.end()) {
            embeddings[word] = std::vector<float>(dimension);
            for (auto& val : embeddings[word]) {
                val = randomFloat();
            }
            normalizeVector(embeddings[word]);
        }
    }

    // Retrieve the embedding vector for a given word
    const std::vector<float>& getEmbedding(const std::string& word) const {
        if (embeddings.find(word) == embeddings.end()) {
            throw std::runtime_error("Word not found in embeddings.");
        }
        return embeddings.at(word);
    }

    // Update embeddings using a context-aware competitive learning algorithm
    void updateWithContext(const std::vector<std::string>& contexts, const std::string& word, const std::string& contextWord, float coOccurrence) {
        if (embeddings.find(word) == embeddings.end() || embeddings.find(contextWord) == embeddings.end()) {
            return; // Skip updates for unknown words
        }

        auto& wordVec = embeddings[word];
        auto& contextVec = embeddings[contextWord];
        std::vector<float> pooledContext = getContextEmbedding(contexts);

        for (size_t i = 0; i < dimension; ++i) {
            float gradient = (pooledContext[i] * wordVec[i] * contextVec[i]) - regularization;
            wordVec[i] += learningRate * gradient;
            contextVec[i] += learningRate * gradient;
        }

        // Add noise and normalize
        addNoise(wordVec, noiseFactor);
        addNoise(contextVec, noiseFactor);
        normalizeVector(wordVec);
        normalizeVector(contextVec);
    }

    // Pool embeddings for a set of contexts
    std::vector<float> getContextEmbedding(const std::vector<std::string>& contexts) const {
        std::vector<float> pooled(dimension, 0.0f);
        for (const auto& ctx : contexts) {
            if (embeddings.find(ctx) != embeddings.end()) {
                for (size_t i = 0; i < dimension; ++i) {
                    pooled[i] += embeddings.at(ctx)[i];
                }
            }
        }
        for (float& val : pooled) {
            val /= contexts.size(); // Average pooling
        }
        return pooled;
    }

    // Competitive update for embeddings
    void competitiveUpdate() {
        for (auto& [word, vec] : embeddings) {
            float maxVal = *std::max_element(vec.begin(), vec.end());
            for (float& val : vec) {
                val = (val == maxVal) ? val + learningRate : val * (1.0f - regularization);
            }
            normalizeVector(vec);
        }
    }

    // Train embeddings for a given co-occurrence dataset
    void train(const std::vector<std::tuple<std::string, std::string, float>>& coOccurrenceData, size_t epochs) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& [word, contextWord, coOccurrence] : coOccurrenceData) {
                updateWithContext({contextWord}, word, contextWord, coOccurrence);
            }
            competitiveUpdate();
        }
    }
    
    void train(const CoOccurrenceData& data, size_t epochs) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& pair : data) {
                updateWithContext(pair.first, pair.second);  // Train on each pair
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " complete." << std::endl;
        }
    }

    void updateWithContext(const std::vector<float>& context, const std::vector<float>& target) {
        for (size_t i = 0; i < dimension; ++i) {
            float hebbianUpdate = learningRate * context[i] * target[i];
            float gradientUpdate = -regularization * weights[i];
            weights[i] += 0.5f * (hebbianUpdate + gradientUpdate);
        }
    }
    

    // Save embeddings to a JSON file
    void save(const std::string& path) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for saving embeddings.");
        }

        file << "{\n"; // Start JSON object
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
        file << "\n}\n"; // End JSON object
    }

    // Load embeddings from a JSON file
    void load(const std::string& path) {
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
    }

    std::string serialize() const {
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

    void deserialize(const std::string& data) {
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
    
} // namespace LM





