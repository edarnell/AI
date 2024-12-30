#ifndef LM_H
#define LM_H

#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

using CoOccurrenceData = std::vector<std::tuple<std::string, std::string, float>>;

class LM {
private:
    std::unordered_map<std::string, std::vector<float>> embeddings; // Word embeddings
    // Helper functions for internal use
    float randomFloat();
    void normalizeVector(std::vector<float>& vec);
    void addNoise(std::vector<float>& vec, float factor);
public:
    int dim=50;               // Dimensionality of embeddings
    float lr=0.01;             // Learning rate for updates
    float reg=0.001;           // Regularization factor for training
    float noise=0.01;              // Noise factor for stochastic updates

    // Constructor
    LM(int dim, float lr, float reg, float noise);

    // Methods for managing words and embeddings
    void addWord(const std::string& word);
    const std::vector<float>& getEmbedding(const std::string& word) const;

    // Context-aware embedding updates
    void updateWithContext(const std::vector<std::string>& contexts,
                           const std::string& word,
                           const std::string& contextWord,
                           float coOccurrence);

    std::vector<float> getContextEmbedding(const std::vector<std::string>& contexts) const;

    // Competitive learning updates
    void competitiveUpdate();

    // Training
    void train(const std::vector<std::tuple<std::string, std::string, float>>& coOccurrenceData, size_t epochs);

    // Serialization and deserialization
    std::string serialize() const;
    void deserialize(const std::string& data);

    // File I/O for embeddings
    void save(const std::string& path) const;
    void load(const std::string& path);
};

#endif // LM_H





