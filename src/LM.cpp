#include "LM.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>

namespace LM {

// Internal storage
namespace {
std::unordered_map<std::string, std::unordered_map<std::string, float>> cooccurrenceMatrix;
std::unordered_map<std::string, std::vector<float>> wordEmbeddings;
size_t embeddingDim = 50;
float learningRate = 0.01f;
float decayRate = 0.001f;

// Utility functions
std::vector<std::string> tokenize(const std::string& text);
std::string toLowerCase(const std::string& text);
std::string stripPunctuation(const std::string& text);
}

// Initializes embeddings and clears the co-occurrence matrix
void initialize(size_t dim) {
    embeddingDim = dim;
    cooccurrenceMatrix.clear();
    wordEmbeddings.clear();
}

// Builds co-occurrence matrix from a dataset
void buildCooccurrenceMatrix(const std::string& datasetPath, size_t windowSize) {
    std::ifstream file(datasetPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset file.");
    }

    std::string line;
    while (std::getline(file, line)) {
        auto tokens = tokenize(stripPunctuation(toLowerCase(line)));

        for (size_t i = 0; i < tokens.size(); ++i) {
            for (size_t j = std::max(0, static_cast<int>(i) - static_cast<int>(windowSize));
                 j < std::min(tokens.size(), i + windowSize + 1);
                 ++j) {
                if (i != j) {
                    cooccurrenceMatrix[tokens[i]][tokens[j]] += 1.0f;
                }
            }
        }
    }

    std::cout << "Co-occurrence matrix built successfully.\n";
}

// Hebbian-inspired weight update
void updateHebbianWeights(const std::string& word, const std::string& contextWord, float coOccurrence) {
    auto& wordVec = wordEmbeddings[word];
    auto& contextVec = wordEmbeddings[contextWord];

    for (size_t i = 0; i < embeddingDim; ++i) {
        // Hebbian learning: Reinforce similarity
        wordVec[i] += learningRate * coOccurrence * contextVec[i];
        contextVec[i] += learningRate * coOccurrence * wordVec[i];

        // Apply decay to prevent runaway reinforcement
        wordVec[i] *= (1.0f - decayRate);
        contextVec[i] *= (1.0f - decayRate);
    }
}

// Trains embeddings using Hebbian updates
void trainHebbianEmbeddings(size_t epochs) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& [word, coOccurrences] : cooccurrenceMatrix) {
            for (const auto& [contextWord, count] : coOccurrences) {
                if (wordEmbeddings.find(contextWord) != wordEmbeddings.end()) {
                    updateHebbianWeights(word, contextWord, log(1 + count));
                }
            }
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed.\n";
    }
}

// Normalizes embeddings to maintain consistency
void normalizeEmbeddings() {
    for (auto& [word, vector] : wordEmbeddings) {
        float magnitude = 0.0f;
        for (float val : vector) {
            magnitude += val * val;
        }
        magnitude = std::sqrt(magnitude);

        if (magnitude > 0) {
            for (float& val : vector) {
                val /= magnitude;
            }
        }
    }
    std::cout << "Embeddings normalized.\n";
}

// Retrieves the embedding for a specific word
std::vector<float> getEmbedding(const std::string& word) {
    if (wordEmbeddings.find(word) != wordEmbeddings.end()) {
        return wordEmbeddings[word];
    }
    return std::vector<float>(embeddingDim, 0.0f); // Return zero vector if word is unknown
}

// Adds a new word to the vocabulary
void addNewWord(const std::string& word) {
    if (wordEmbeddings.find(word) == wordEmbeddings.end()) {
        std::vector<float> newVector(embeddingDim, 0.1f); // Initialize with small values
        wordEmbeddings[word] = newVector;
        std::cout << "New word added to vocabulary: " << word << "\n";
    }
}

// Saves embeddings to a file
void saveEmbeddings(const std::string& outputPath) {
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving embeddings.");
    }

    for (const auto& [word, embedding] : wordEmbeddings) {
        file << word;
        for (float value : embedding) {
            file << " " << value;
        }
        file << "\n";
    }

    std::cout << "Embeddings saved to " << outputPath << ".\n";
}

// Loads embeddings from a file
void loadEmbeddings(const std::string& inputPath) {
    std::ifstream file(inputPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading embeddings.");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string word;
        stream >> word;

        std::vector<float> embedding;
        float value;
        while (stream >> value) {
            embedding.push_back(value);
        }

        wordEmbeddings[word] = embedding;
    }

    std::cout << "Embeddings loaded from " << inputPath << ".\n";
}

// Text processing utilities
namespace {
std::vector<std::string> tokenize(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string toLowerCase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return result;
}

std::string stripPunctuation(const std::string& text) {
    std::string result;
    std::remove_copy_if(text.begin(), text.end(), std::back_inserter(result), [](unsigned char c) {
        return std::ispunct(c);
    });
    return result;
}
} // namespace

} // namespace LM

