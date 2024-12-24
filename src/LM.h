#ifndef LM_H
#define LM_H

#include <string>
#include <vector>
#include <unordered_map>

namespace LM {

    class Model {
    private:
        std::unordered_map<std::string, std::vector<float>> embeddings; 
        size_t dimension;
        float learningRate;
        float regularization;
        float noiseFactor;

        float randomFloat();
        void normalizeVector(std::vector<float>& vec);
        void addNoise(std::vector<float>& vec, float factor);

    public:
        Model(size_t dim = 50, float lr = 0.01, float reg = 0.001, float noise = 0.01);

        void addWord(const std::string& word);
        const std::vector<float>& getEmbedding(const std::string& word) const;

        void updateWithContext(const std::vector<std::string>& contexts,
                               const std::string& word,
                               const std::string& contextWord,
                               float coOccurrence);

        std::vector<float> getContextEmbedding(const std::vector<std::string>& contexts) const;
        void competitiveUpdate();
        void train(const std::vector<std::tuple<std::string, std::string, float>>& coOccurrenceData, size_t epochs);

        void save(const std::string& path) const;
        void load(const std::string& path);
    };

} // namespace LM

#endif // LM_H



