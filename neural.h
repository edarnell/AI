#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include <string>

// Structure representing a neuron
struct Neuron {
    double activation = 0.0; // Activation value of the neuron
};

class NeuralNet {
public:
    // Constructor to initialize the network structure with the given layer sizes
    NeuralNet(const std::vector<int>& layers);

    // Adjust the size of the input layer
    void adjustInputLayerSize(int newSize);

    // Method to get the size of the input layer (new addition)
    int getInputNeuronCount() const;

    // Method to perform forward propagation with input data
    void forwardPropagate(const std::vector<double>& input);

    // Method to train the network using a batch of articles
    void train(const std::vector<std::string>& articles);

    // Method to preprocess article content for input into the network
    std::vector<double> preprocessArticle(const std::string& content);

    // Method to extract features from an article
    std::vector<double> extractFeatures(const std::string& article);

private:
    std::vector<std::vector<Neuron>> layers; // Layers of neurons
    std::vector<std::vector<std::vector<double>>> weights; // Weights between layers

    // Helper function for activation
    double activationFunction(double x);
};

#endif // NEURAL_H

