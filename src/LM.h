#ifndef LM_H
#define LM_H

#include <string>
#include <unordered_map>
#include <vector>

namespace LM {

// Initializes embeddings and co-occurrence matrix
void initialize(size_t embeddingDim);

// Builds co-occurrence matrix from a dataset
void buildCooccurrenceMatrix(const std::string& datasetPath, size_t windowSize);

// Trains embeddings using Hebbian-inspired updates
void trainHebbianEmbeddings(size_t epochs);

// Normalizes embeddings to maintain consistency
void normalizeEmbeddings();

// Retrieves the embedding for a specific word
std::vector<float> getEmbedding(const std::string& word);

// Adds a new word to the vocabulary
void addNewWord(const std::string& word);

// Saves embeddings to a file
void saveEmbeddings(const std::string& outputPath);

// Loads embeddings from a file
void loadEmbeddings(const std::string& inputPath);

// Adaptive training with dynamically adjusted parameters
void trainWithAdaptiveParameters(const std::string& datasetPath, size_t epochs);

// Expands embedding dimensions dynamically
void expandEmbeddingDimensions(size_t newDim);

// Dynamically initializes parameters based on dataset properties
void initializeDynamicParameters(const std::string& datasetPath);

} // namespace LM

#endif // LM_H
