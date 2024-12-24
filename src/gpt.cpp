#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "N3R.h"
#include "LM.h"
#include "utils.h"

namespace GPT {
    // Global variables for the conversational model
    N3R::NNet nnet;
    // Default file path for JSON conversation data
    const std::string fGPT = "data/conversations.json";
    std::string sha; // Track last trained state
    
    // Train or retrain the model based on data changes
    bool init(LM::Model& model, const std::string& f) {
        std::string h=sha256(fGPT);
        load(f);
        if (h!=sha) {
            std::cout << "New data detected. Incremental training starting...\n";
            loadJSON(fGPT);  // Load new conversations
            train(10);            // Default incremental training
            sha = h;  // Update hash
            std::cout << "Incremental training complete.\n";
            return true;
        } else {
            std::cout << "No data updates. Model is up to date.\n";
            return false;
        }
    }

    // Load JSON conversation data (manual parsing)
    void loadJSON(const std::string& filePath = fGPT) {
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

        std::cout << "Loaded " << convoCount << " conversations from " << filePath << ".\n";
    }
    
    // Incrementally update the model with new user feedback
    void iTrain(const std::string& userInput, const std::string& correctResponse) {
        nnet.addN(userInput, "input", 1.0f);
        nnet.addN(correctResponse, "output", 0.0f);
        nnet.addS(userInput, correctResponse, 0.5f);
        nnet.fwd();  // Update the network
        nnet.validate();
        std::cout << "Model updated incrementally.\n";
    }

    // Train conversational embeddings
    void train(size_t epochs) {
        for (size_t e = 0; e < epochs; ++e) {
            nnet.fwd();  // Forward propagate through the network
            std::cout << "Training epoch " << epoch + 1 << "/" << epochs << std::endl;
            // Optionally introduce noise or variability for stochastic updates
            nnet.addWeightNoise(0.01f);
            adjustParameters(epoch);
        }

        // Normalize the final embeddings for consistency
        nnet.validate();
        std::cout << "Training completed over " << epochs << " epochs.\n";
    }
    
    void adjustParameters(size_t epoch) {
        learningRate = std::max(0.001f, learningRate * 0.95f);  // Decay learning rate
        regularization += 0.0001f * epoch;  // Increment regularization
    }

    // Generate a response based on user input
    std::string genResp(const std::string& userInput) {
        // Forward propagate user input through the network
        auto contextEmbedding = embeddingModel.getContextEmbedding({userInput});
        nnet.addN(userInput, "input", 1.0f);  // Ensure user input is in the network

        // Find the strongest connected response
        std::string bestResponse;
        float maxWeight = -1.0f;

        for (const auto& syn : nnet.synapses) {
            if (syn.src == userInput && syn.weight > maxWeight) {
                maxWeight = syn.weight;
                bestResponse = syn.dest;
            }
        }

        return bestResponse.empty() ? "I don't know yet." : bestResponse;
    }

    // Save the model to disk
    void save(const std::string& filePath) {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Unable to save model to file.");
        }
        nnet.print();  // Save the network's current state
        std::cout << "Model saved to " << filePath << ".\n";
    }

    // Load the model from disk
    void load(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Unable to load model from file.");
        }
        // Placeholder for loading logic
        std::cout << "Loading model from " << filePath << "...\n";
    }

    // Interactive prompt for testing the conversation system
    void interactivePrompt() {
        std::string userInput;
        std::cout << "Start chatting with Xi! Type 'exit' to quit.\n";
        while (true) {
            std::cout << "You: ";
            std::getline(std::cin, userInput);
            if (userInput == "exit") break;

            std::string response = generateResponse(userInput);
            std::cout << "Xi: " << response << "\n";
        }
    }

}  // namespace GPT
