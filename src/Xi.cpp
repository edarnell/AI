#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <bzlib.h>
#include "utils.h"  // Utility functions
#include "LM.h"     // Model 
#include "N3R.h"    // Neural Network logic
#include "Xi.h" 


namespace Xi {

    // Global state for training and model management
    LM::Model model(100, 0.01, 0.001);
    const std::string fM = "data/model.bz2";
    const std::string fGPT = "data/conversations.json";

    N3R::NNet nnet;
    std::string sha; // Track last trained state

    void loadModel(const std::string& f) {
        std::cout << "Loading model: " << f << std::endl;
        // Load and train on updated data
        if (sha256(fGPT) != sha) {
            load(f);
            std::cout << "New data detected. Incremental training starting...\n";
            loadJSON();
            train(10);
            sha = sha256(fGPT);
            save(f);
            std::cout << "Training complete. Model updated." << std::endl;
        } else {
            std::cout << "No data updates. Model is up to date.\n";
        }
    }
    
    void load(const std::string& filePath) {
        BZFILE* file = BZ2_bzopen(filePath.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("Unable to open model file for reading: " + filePath);
        }

        constexpr int BUFFER_SIZE = 4096;
        char buffer[BUFFER_SIZE];
        std::string modelData;
        int bytesRead;

        while ((bytesRead = BZ2_bzread(file, buffer, BUFFER_SIZE)) > 0) {
            modelData.append(buffer, bytesRead);
        }

        BZ2_bzclose(file);

        // Deserialize model data (Assuming the model supports a deserialize method)
        model.deserialize(modelData);
        std::cout << "Model loaded successfully from " << filePath << std::endl;
    }
    
    void save(const std::string& filePath) {
        BZFILE* file = BZ2_bzopen(filePath.c_str(), "wb");
        if (!file) {
            throw std::runtime_error("Unable to open model file for writing: " + filePath);
        }

        // Serialize model data (Assuming the model supports a serialize method)
        std::string modelData = model.serialize();

        if (BZ2_bzwrite(file, modelData.data(), modelData.size()) < 0) {
            BZ2_bzclose(file);
            throw std::runtime_error("Error writing model data to file: " + filePath);
        }

        BZ2_bzclose(file);
        std::cout << "Model saved successfully to " << filePath << std::endl;
    }

    void loadJSON() {
        std::unordered_map<std::string, std::string> nodeData;
        auto parentChildMap = Utils::chat(fGPT, nodeData);

        size_t convoCount = 0;

        // Traverse the parent-child map to reconstruct multi-turn conversations
        for (const auto& [parent, children] : parentChildMap) {
            std::string parentContent = nodeData[parent];

            for (const auto& child : children) {
                std::string childContent = nodeData[child];

                // Example: Add parent-child relationship to the neural net
                nnet.addN(parentContent, "input", 1.0f);
                nnet.addN(childContent, "output", 0.0f);
                nnet.addS(parentContent, childContent, 0.5f);

                convoCount++;
            }
        }

        std::cout << "Processed " << convoCount << " conversational turns from chat data.\n";
    }

    void train(size_t epochs) {
        for (size_t e = 0; e < epochs; ++e) {
            nnet.fwd();
            std::cout << "Training epoch " << e + 1 << "/" << epochs << std::endl;
            nnet.addWeightNoise(0.01f);
            adjustParameters(e);
        }

        nnet.validate();
        std::cout << "Training completed over " << epochs << " epochs.\n";
    }

    void adjustParameters(size_t epoch) {
        model.learningRate = std::max(0.001f, model.learningRate * 0.95f);
        model.regularization += 0.0001f * epoch;
    }

    std::string generateResponse(const std::string& userInput) {
        auto contextEmbedding = model.getContextEmbedding({userInput});
        nnet.addN(userInput, "input", 1.0f);

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

    void saveConversation(const std::string& topic, const std::vector<std::pair<int64_t, std::string>>& newMessages) {
        Utils::appendToBzip2(fGPT, topic, newMessages);
        std::cout << "Saved conversation to topic: " << topic << std::endl;
    }

}  // namespace Xi
