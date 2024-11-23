#include "neural.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <regex> // For parsing XML-like structures

// Constructor for initializing the neural network with layers
NeuralNet::NeuralNet(const std::vector<int>& layers) {
    for (size_t i = 0; i < layers.size(); ++i) {
        this->layers.push_back(std::vector<Neuron>(layers[i]));
        if (i > 0) {
            weights.push_back(std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i - 1])));
            for (auto& neuronWeights : weights.back()) {
                std::generate(neuronWeights.begin(), neuronWeights.end(), []() { return (rand() / double(RAND_MAX)) * 2.0 - 1.0; });
            }
        }
    }
}

int NeuralNet::getInputNeuronCount() const {
    if (layers.empty()) {
        throw std::runtime_error("Neural network has no layers defined.");
    }
    return layers[0].size();
}

void NeuralNet::adjustInputLayerSize(int newSize) {
    if (!layers.empty()) {
        layers[0] = std::vector<Neuron>(newSize); // Resize the input layer
    }
}

// Activation function (sigmoid)
double NeuralNet::activationFunction(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Forward propagation
void NeuralNet::forwardPropagate(const std::vector<double>& input) {
    if (input.size() != layers[0].size()) {
        throw std::invalid_argument("Input size does not match the number of input neurons.");
    }

    // Set input layer activations
    for (size_t i = 0; i < input.size(); ++i) {
        layers[0][i].activation = input[i];
    }

    // Propagate through hidden layers and output layer
    for (size_t l = 1; l < layers.size(); ++l) {
        for (size_t j = 0; j < layers[l].size(); ++j) {
            double totalInput = 0.0;
            for (size_t k = 0; k < layers[l - 1].size(); ++k) {
                totalInput += layers[l - 1][k].activation * weights[l - 1][j][k];
            }
            layers[l][j].activation = activationFunction(totalInput);
        }
    }
}

// Train method with integration for article-based training
void NeuralNet::train(const std::vector<std::string>& articles) {
    for (const auto& article : articles) {
        // Tokenize and preprocess each article
        std::vector<double> processedInput = preprocessArticle(article);

        // Extract and append features
        std::vector<double> features = extractFeatures(article);
        processedInput.insert(processedInput.end(), features.begin(), features.end());

        // Adjust input layer size if necessary
        if (processedInput.size() != layers[0].size()) {
            std::cerr << "Adjusting input layer size from " << layers[0].size()
                      << " to " << processedInput.size() << " neurons.\n";
            adjustInputLayerSize(processedInput.size());
        }

        // Forward propagate the processed input
        forwardPropagate(processedInput);

        // Debugging output
        std::cout << "Training completed for article with initial text: " 
                  << article.substr(0, 50) << "...\n";
        std::cout << "Extracted Features: ";
        for (const auto& feature : features) {
            std::cout << feature << " ";
        }
        std::cout << "\n";
    }
}

// Preprocessing logic to convert article content into network input
std::vector<double> NeuralNet::preprocessArticle(const std::string& content) {
    std::vector<double> processedInput;
    for (char c : content) {
        processedInput.push_back(static_cast<double>(c) / 255.0); // Normalize ASCII values
    }
    return processedInput;
}

// Method to extract specific features (e.g., word definitions, disambiguation, categories)
std::vector<double> NeuralNet::extractFeatures(const std::string& article) {
    std::vector<double> features;

    // Check for disambiguation
    if (article.find("<disambiguation>") != std::string::npos) {
        features.push_back(1.0); // Example: flag for disambiguation
    } else {
        features.push_back(0.0);
    }

    // Extract categories
    std::regex categoryRegex("\\[\\[Category:(.*?)\\]\\]");
    auto categoriesBegin = std::sregex_iterator(article.begin(), article.end(), categoryRegex);
    int categoryCount = std::distance(categoriesBegin, std::sregex_iterator());
    features.push_back(static_cast<double>(categoryCount)); // Example: category count

    // Infobox presence
    std::regex infoboxRegex("\\{\\{Infobox((?:.|\\n)*?)\\}\\}");

    if (std::regex_search(article, infoboxRegex)) {
        features.push_back(1.0); // Flag for infobox presence
    } else {
        features.push_back(0.0);
    }

    // Section headings count
    std::regex headingRegex("==([^=]+)==");
    auto headingsBegin = std::sregex_iterator(article.begin(), article.end(), headingRegex);
    int headingCount = std::distance(headingsBegin, std::sregex_iterator());
    features.push_back(static_cast<double>(headingCount)); // Number of headings

    // Link count
    std::regex linkRegex("\\[\\[(.*?)\\]\\]");
    auto linksBegin = std::sregex_iterator(article.begin(), article.end(), linkRegex);
    int linkCount = std::distance(linksBegin, std::sregex_iterator());
    features.push_back(static_cast<double>(linkCount)); // Number of links

    return features;
}
