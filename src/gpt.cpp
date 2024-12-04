#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include "N3R.h"  // Neural Network and Logic Validation
#include "LM.h"   // Language Model

class GPTSystem {
    N3R::NNet neuralNet;          // Neural Network for logical relations
    LM::LanguageModel languageModel; // Language Model for embeddings and co-occurrence

public:
    // Load and train from a JSON-like dataset
    void train(const std::string& filePath = "data/gpt.json") {
        std::ifstream inputFile(filePath);
        if (!inputFile.is_open()) {
            throw std::runtime_error("Error: Cannot open file " + filePath);
        }

        std::string line;
        std::string content = "";
        while (std::getline(inputFile, line)) {
            content += line;
        }
        inputFile.close();

        auto dataEntries = parseJSON(content);
        for (const auto& entry : dataEntries) {
            std::string userInput = entry.first;
            std::string assistantOutput = entry.second;

            // Process the input-output pair
            processKnowledgePair(userInput, assistantOutput);
        }

        languageModel.norm(); // Normalize embeddings
        std::cout << "Training completed successfully.\n";
        performSelfAssessment();
    }

private:
    // Parse JSON structure with mixed content
    std::vector<std::pair<std::string, std::string>> parseJSON(const std::string& content) {
        std::vector<std::pair<std::string, std::string>> dataEntries;
        size_t start = 0;

        while ((start = content.find("\"role\": \"user\"", start)) != std::string::npos) {
            size_t userContentStart = content.find("\"content\": ", start) + 11;
            size_t userContentEnd = findContentEnd(content, userContentStart);
            std::string userInput = content.substr(userContentStart, userContentEnd - userContentStart);
            sanitizeContent(userInput);

            size_t assistantStart = content.find("\"role\": \"assistant\"", userContentEnd);
            if (assistantStart == std::string::npos) break;

            size_t assistantContentStart = content.find("\"content\": ", assistantStart) + 11;
            size_t assistantContentEnd = findContentEnd(content, assistantContentStart);
            std::string assistantOutput = content.substr(assistantContentStart, assistantContentEnd - assistantContentStart);
            sanitizeContent(assistantOutput);

            dataEntries.emplace_back(userInput, assistantOutput);
            start = assistantContentEnd;
        }

        return dataEntries;
    }

    // Determine the end of content, handling structured data or plain text
    size_t findContentEnd(const std::string& content, size_t start) {
        if (content[start] == '{' || content[start] == '[') {
            // Structured data block
            int openBrackets = 1;
            size_t pos = start + 1;
            while (openBrackets > 0 && pos < content.size()) {
                if (content[pos] == '{' || content[pos] == '[') openBrackets++;
                if (content[pos] == '}' || content[pos] == ']') openBrackets--;
                pos++;
            }
            return pos;
        } else {
            // Plain text block
            return content.find("\"", start + 1);
        }
    }

    // Clean and format content
    void sanitizeContent(std::string& content) {
        // Remove enclosing quotes
        if (!content.empty() && content[0] == '"') content.erase(0, 1);
        if (!content.empty() && content.back() == '"') content.pop_back();

        // Handle structured data: Convert to readable string if necessary
        if (content.find("{") == 0 || content.find("[") == 0) {
            content = parseStructuredData(content);
        }
    }

    // Parse structured data (JSON-like) into a readable string format
    std::string parseStructuredData(const std::string& content) {
        std::ostringstream parsedContent;

        if (content[0] == '{') {
            // Convert object to key-value pairs
            parsedContent << "Key-Value Data: ";
            size_t start = 1; // Skip opening '{'
            while (start < content.size() && content[start] != '}') {
                size_t keyStart = content.find("\"", start) + 1;
                size_t keyEnd = content.find("\"", keyStart);
                std::string key = content.substr(keyStart, keyEnd - keyStart);

                size_t valueStart = content.find(":", keyEnd) + 1;
                size_t valueEnd = content.find(",", valueStart);
                if (valueEnd == std::string::npos) valueEnd = content.find("}", valueStart);
                std::string value = content.substr(valueStart, valueEnd - valueStart);
                sanitizeContent(value); // Clean value

                parsedContent << key << "=" << value << "; ";
                start = valueEnd + 1;
            }
        } else if (content[0] == '[') {
            // Convert array to CSV-like format
            parsedContent << "Array Data: ";
            size_t start = 1; // Skip opening '['
            while (start < content.size() && content[start] != ']') {
                size_t valueStart = content.find_first_not_of(" \n\r\t", start);
                size_t valueEnd = content.find(",", valueStart);
                if (valueEnd == std::string::npos) valueEnd = content.find("]", valueStart);
                std::string value = content.substr(valueStart, valueEnd - valueStart);
                sanitizeContent(value); // Clean value

                parsedContent << value << ", ";
                start = valueEnd + 1;
            }
        }

        std::string result = parsedContent.str();
        if (result.back() == ' ') result.pop_back(); // Remove trailing space
        if (result.back() == ',') result.pop_back(); // Remove trailing comma
        return result;
    }

    // Process user input and assistant output as a knowledge pair
    void processKnowledgePair(const std::string& userInput, const std::string& assistantOutput) {
        neuralNet.addN(userInput, 1.0);
        neuralNet.addN(assistantOutput, 0.5);
        neuralNet.addS(userInput, assistantOutput, 0.8, 1.0);

        languageModel.addWrd(userInput); // Add words to LM
        languageModel.addWrd(assistantOutput);
    }

    // Perform self-assessment to identify gaps
    void performSelfAssessment() {
        std::cout << "Performing self-assessment...\n";
        neuralNet.validate(N3R::Model, 0.05, 0.2, 0.999, 0.001); // Comprehensive validation
        std::cout << "Model validation complete.\n";
    }
};

int main(int argc, char* argv[]) {
    try {
        GPTSystem system;
        std::string filePath = (argc > 1) ? argv[1] : "data/gpt.json";
        std::cout << "Training from file: " << filePath << std::endl;
        system.train(filePath);
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
