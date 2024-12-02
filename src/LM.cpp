#include "LM.h"
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>

namespace LM {

namespace {

// Internal model parameters
size_t inputSize, hiddenSize, outputSize;
std::vector<std::vector<float>> weightsInputHidden;
std::vector<std::vector<float>> weightsHiddenOutput;
std::vector<float> biasHidden;
std::vector<float> biasOutput;

// Internal data
std::vector<std::vector<float>> trainingData;
std::vector<size_t> trainingLabels;
std::unordered_set<std::string> vocabulary;
std::unordered_map<std::string, std::vector<float>> userPreferences;

// Activation functions
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float sigmoidDerivative(float x) { return x * (1.0f - x); }

// Random initialization for weights
float randomWeight() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    return dist(rng);
}

// Initializes weights and biases
void initializeWeights() {
    weightsInputHidden.assign(inputSize, std::vector<float>(hiddenSize, randomWeight()));
    weightsHiddenOutput.assign(hiddenSize, std::vector<float>(outputSize, randomWeight()));
    biasHidden.assign(hiddenSize, randomWeight());
    biasOutput.assign(outputSize, randomWeight());
}

// Computes hidden layer activations
std::vector<float> computeHiddenLayer(const std::vector<float>& input) {
    std::vector<float> hidden(hiddenSize, 0.0f);
    for (size_t i = 0; i < hiddenSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            hidden[i] += input[j] * weightsInputHidden[j][i];
        }
        hidden[i] += biasHidden[i];
        hidden[i] = sigmoid(hidden[i]);
    }
    return hidden;
}

// Computes output layer activations
std::vector<float> computeOutputLayer(const std::vector<float>& hidden) {
    std::vector<float> output(outputSize, 0.0f);
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < hiddenSize; ++j) {
            output[i] += hidden[j] * weightsHiddenOutput[j][i];
        }
        output[i] += biasOutput[i];
        output[i] = sigmoid(output[i]);
    }
    return output;
}

// Updates weights using Hebbian learning
void updateWeightsHebbian(const std::vector<float>& input, const std::vector<float>& hidden) {
    for (size_t i = 0; i < inputSize; ++i) {
        for (size_t j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] += 0.01f * input[i] * hidden[j];
        }
    }
}

// Applies global optimization (evolutionary strategy)
void optimizeWeightsEvolutionary() {
    for (auto& row : weightsInputHidden) {
        for (auto& weight : row) {
            weight += randomWeight() * 0.01f;
        }
    }
    for (auto& row : weightsHiddenOutput) {
        for (auto& weight : row) {
            weight += randomWeight() * 0.01f;
        }
    }
}

} // namespace

void initialize(size_t inSize, size_t hidSize, size_t outSize) {
    inputSize = inSize;
    hiddenSize = hidSize;
    outputSize = outSize;
    initializeWeights();
}

void addTrainingSample(const std::vector<float>& input, size_t labelIndex) {
    if (input.size() != inputSize) {
        throw std::invalid_argument("Input size does not match model input size.");
    }
    trainingData.push_back(input);
    trainingLabels.push_back(labelIndex);
}

void train(const std::vector<std::vector<float>>& data, const std::vector<size_t>& labels, size_t epochs) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t sample = 0; sample < data.size(); ++sample) {
            const auto& input = data[sample];
            auto hidden = computeHiddenLayer(input);
            updateWeightsHebbian(input, hidden);

            if (epoch % 10 == 0) {
                optimizeWeightsEvolutionary();
            }
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed.\n";
    }
}

void updateModel(const std::vector<float>& input, size_t labelIndex) {
    addTrainingSample(input, labelIndex);
    auto hidden = computeHiddenLayer(input);
    updateWeightsHebbian(input, hidden);
    optimizeWeightsEvolutionary();
}

void updateVocabulary(const std::string& text) {
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        if (vocabulary.insert(word).second) {
            std::cout << "New word added to vocabulary: " << word << "\n";
        }
    }
}

void integrateFeedback(const std::string& feedback, const std::vector<float>& input) {
    if (feedback.find("great") != std::string::npos) {
        auto hidden = computeHiddenLayer(input);
        updateWeightsHebbian(input, hidden);
    } else if (feedback.find("error") != std::string::npos) {
        auto hidden = computeHiddenLayer(input);
        for (size_t i = 0; i < inputSize; ++i) {
            for (size_t j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] -= 0.01f * input[i] * hidden[j];
            }
        }
    }
}

std::vector<float> infer(const std::vector<float>& input) {
    auto hidden = computeHiddenLayer(input);
    return computeOutputLayer(hidden);
}

std::unordered_map<std::string, std::string> classify(const std::vector<float>& input) {
    auto probabilities = infer(input);
    size_t maxIndex = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

    std::unordered_map<std::string, std::string> result;
    result["polarity"] = (maxIndex == 0 ? "negative" : (maxIndex == 1 ? "neutral" : "positive"));
    result["intensity"] = (probabilities[maxIndex] > 0.7 ? "high" : (probabilities[maxIndex] > 0.4 ? "medium" : "low"));
    return result;
}

void updateUserPreferences(const std::string& userId, const std::vector<float>& preferences) {
    userPreferences[userId] = preferences;
}

std::vector<float> personalizeInference(const std::vector<float>& input, const std::string& userId) {
    auto output = infer(input);
    if (userPreferences.find(userId) != userPreferences.end()) {
        auto& preferences = userPreferences[userId];
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += preferences[i];
        }
    }
    return output;
}

std::vector<float> preprocessText(const std::string& text) {
    std::vector<float> vectorized(inputSize, 0.0f);
    for (size_t i = 0; i < text.size() && i < inputSize; ++i) {
        vectorized[i] = static_cast<float>(text[i]) / 255.0f;
    }
    return vectorized;
}

} // namespace LM
