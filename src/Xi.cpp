#include <iostream>
#include <string>
#include <chrono>
#include "gpt.h"  // GPT conversation logic
#include "utils.h"  // Utility functions
#include "LM.h"     // Model
#include "N3R.h"    // Neural Network logic

namespace Xi {

    // Global state for training and model management
    LM::Model model(100, 0.01, 0.001);
    const std::string fM = "data/model.json";
    
    void load(const std::string& f=fM) {
        std::cout << "Loading model: " << f << std::endl;
        // Load and train on updated data
        if (GPT::init(model,f)) {// Perform incremental training
            // Save the updated model
            GPT::save(f);
            std::cout << "Training complete. Model updated." << std::endl;
        }
    }
    
    void summarizeTopic(const std::string& topic) {
        try {
            auto messages = Utils::readTopic(conversationFile, topic);
            std::cout << "Summary of topic '" << topic << "':\n";
            for (const auto& [timestamp, message] : messages) {
                std::cout << "[" << Utils::formatXiTime(timestamp) << "] " << message << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "No previous conversations for topic: " << topic << std::endl;
        }
    }

    saveConversation(const std::string& topic, const std::vector<std::pair<int64_t, std::string>>& newMessages) {
        appendToBzip2(conversationFile, topic, newMessages);
        std::cout << "Saved conversation to topic: " << topic << std::endl;
    }
        
    void mainLoop() {
        trainOrRetrain();  // Ensure model is trained at startup
        
        std::string command;
        std::string currentTopic = "general";
        std::vector<std::pair<int64_t, std::string>> currentMessages;
        std::cout << "Welcome to Xi! Current Topic: " << currentTopic << std::endl;

        while (true) {
            std::cout << "Xi> ";
            std::getline(std::cin, command);
            // Exit command
            if (command == "exit" || command == "quit") {
                if (!currentMessages.empty()) {
                    saveConversation(currentTopic, currentMessages);
                    currentMessages.clear();
                }
                saveModel(fM); // Ensure the latest state is stored
                std::cout << "Goodbye!" << std::endl;
                break;
            }
            // Topic switch
            if (command.starts_with("topic:")) {
                std::string newTopic = command.substr(6);
                Utils::trim(newTopic);
                if (newTopic.empty()) {
                    std::cout << "Please specify a topic name." << std::endl;
                    continue;
                }
                if (!currentMessages.empty()) {
                    saveConversation(currentTopic, currentMessages);
                    currentMessages.clear();
                }
                currentTopic = newTopic;
                std::cout << "Switched to topic: " << currentTopic << std::endl;
                summarizeTopic(currentTopic);
                continue;
            }

            // Handle general input
            if (!command.empty()) {
                int64_t timestamp = Utils::currentXiTime();
                currentMessages.emplace_back(timestamp, command);

                // Echo back for clarity
                std::cout << "You: " << command << std::endl;
            }
        }
    }
}  // namespace Xi

int main() {
    Xi::mainLoop();
    return 0;
}
