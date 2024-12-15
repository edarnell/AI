
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "N3R.h"
#include "LM.h"

namespace GPT {

// Global variables for the conversational model
N3R::NNet nnet;
LM::EmbeddingModel embeddingModel;

// Default file path for JSON conversation data
const std::string DEFAULT_JSON_PATH = "data/conversations.json";

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" 	

");
    size_t last = str.find_last_not_of(" 	

");
    return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, last - first + 1);
}

// Basic JSON parsing utility (minimal implementation)
std::unordered_map<std::string, std::string> parseJSONLine(const std::string& line) {
    std::unordered_map<std::string, std::string> parsed;
    std::istringstream stream(line);
    std::string key, value;
    while (std::getline(stream, key, ':') && std::getline(stream, value, ',')) {
        key = trim(key);
        value = trim(value);
        if (!key.empty() && !value.empty()) {
            key = key.substr(1, key.size() - 2); // Remove quotes
            value = value.substr(1, value.size() - 2); // Remove quotes
            parsed[key] = value;
        }
    }
    return parsed;
}

// Load JSON conversation data (manual parsing)
void loadJSONData(const std::string& filePath = DEFAULT_JSON_PATH) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open JSON file at " + filePath);
    }

    std::string line;
    size_t convoCount = 0;
    while (std::getline(file, line)) {
        auto convo = parseJSONLine(line);
        if (convo.count("user_input") && convo.count("system_response")) {
            const std::string userInput = convo["user_input"];
            const std::string systemResponse = convo["system_response"];

            // Add user input and system response as nodes to the network
            nnet.addN(userInput, "input", 1.0f);
            nnet.addN(systemResponse, "output", 0.0f);

            // Create relationships between inputs and responses
            nnet.addS(userInput, systemResponse, 0.5f); // Initial weight
            convoCount++;
        }
    }

    std::cout << "Loaded " << convoCount << " conversations from " << filePath << ".
";
}

// Train conversational embeddings
void trainModel(size_t epochs) {
    for (size_t e = 0; e < epochs; ++e) {
        nnet.fwd();  // Forward propagate through the network

        // Optionally introduce noise or variability for stochastic updates
        nnet.addWeightNoise(0.01f);
    }

    // Normalize the final embeddings for consistency
    nnet.validate();
    std::cout << "Training completed over " << epochs << " epochs.
";
}

// Generate a response based on user input
std::string generateResponse(const std::string& userInput) {
    // Forward propagate user input through the network
    auto contextEmbedding = LM::getContextEmbedding({userInput});
    nnet.addN(userInput, "input", 1.0f);  // Ensure user input is in the network

    // Find the strongest connected response
    std::string bestResponse;
    float maxWeight = -1.0f;

    for (const auto& syn : nnet.getSynapses()) {
        if (syn.src == userInput && syn.weight > maxWeight) {
            maxWeight = syn.weight;
            bestResponse = syn.dest;
        }
    }

    return bestResponse.empty() ? "I don't know yet." : bestResponse;
}

// Save the model to disk
void saveModel(const std::string& filePath) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to save model to file.");
    }
    nnet.print();  // Save the network's current state
    std::cout << "Model saved to " << filePath << ".
";
}

// Load the model from disk
void loadModel(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to load model from file.");
    }
    // Placeholder for loading logic
    std::cout << "Loading model from " << filePath << "...
";
}

// Interactive prompt for testing the conversation system
void interactivePrompt() {
    std::string userInput;
    std::cout << "Start chatting with Xi! Type 'exit' to quit.
";
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, userInput);
        if (userInput == "exit") break;

        std::string response = generateResponse(userInput);
        std::cout << "Xi: " << response << "
";
    }
}

}  // namespace GPT
