#ifndef XI_H
#define XI_H

#include <string>
#include <vector>
#include <utility>

namespace Xi {
    // Initialize and load the model
    void loadModel(const std::string& f = "data/model.bz2");
    void train(size_t epochs); // Train the model
    // Generate a response based on user input
    std::string generateResponse(const std::string& userInput);
    // Save a conversation topic
    void saveConversation(const std::string& topic, const std::vector<std::pair<int64_t, std::string>>& newMessages);
    void loadJSON(); // Load JSON conversation data
    void load(const std::string& filePath); // Load model from a bzip2 file
    void save(const std::string& filePath); // Save model to a bzip2 file
}

#endif // XI_H
