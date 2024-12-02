#ifndef LM_H
#define LM_H

#include <string>
#include <vector>
#include <unordered_map>

namespace LM {

// Initializes the language model with specific dimensions
void initialize(size_t inputSize, size_t hiddenSize, size_t outputSize);

// Adds a training sample for incremental updates
void addTrainingSample(const std::vector<float>& input, size_t labelIndex);

// Trains the model on all accumulated data
void train(const std::vector<std::vector<float>>& data, const std::vector<size_t>& labels, size_t epochs);

// Dynamically updates the model with a single sample
void updateModel(const std::vector<float>& input, size_t labelIndex);

// Expands vocabulary by adding new words dynamically
void updateVocabulary(const std::string& text);

// Integrates user feedback for refining the model
void integrateFeedback(const std::string& feedback, const std::vector<float>& input);

// Runs inference and returns probabilities for each output class
std::vector<float> infer(const std::vector<float>& input);

// Maps probabilities to human-readable classifications
std::unordered_map<std::string, std::string> classify(const std::vector<float>& input);

// Updates user-specific preferences based on interactions
void updateUserPreferences(const std::string& userId, const std::vector<float>& preferences);

// Generates personalized inference based on user-specific preferences
std::vector<float> personalizeInference(const std::vector<float>& input, const std::string& userId);

// Preprocesses text into vectorized representation
std::vector<float> preprocessText(const std::string& text);

} // namespace LM

#endif // LM_H
