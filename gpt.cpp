#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <map>

// Neuron Structure
struct Neuron {
    double state;
    double threshold;
    std::string content;

    Neuron(double t = 1.0, const std::string& c = "") : state(0.0), threshold(t), content(c) {}
};

// Synapse Structure
struct Synapse {
    int from;
    int to;
    double weight;
    double delay;
    double distance;

    Synapse(int f, int t, double w, double d, double dist)
        : from(f), to(t), weight(w), delay(d), distance(dist) {}
};

// Neural Network Class
class Synaptic3RNet {
private:
    std::vector<Neuron> neurons;
    std::vector<Synapse> synapses;

    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Basic JSON-like parsing function
    void parseJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << "\n";
            return;
        }

        std::string line, content, conversationTitle, userInput;
        int id = 0, parent = -1;
        int conversationCount = 0;
        bool inConversation = false;

        std::map<std::string, int> lastSeen; // Track last seen conversation index for each topic

        while (std::getline(file, line)) {
            try {
                if (line.find("\"conversation_title\"") != std::string::npos) {
                    size_t start = line.find(":") + 2; // Skip ": "
                    size_t end = line.rfind("\"");
                    conversationTitle = line.substr(start, end - start);
                }

                if (line.find("\"user_input\"") != std::string::npos && !inConversation) {
                    size_t start = line.find(":") + 2; // Skip ": "
                    size_t end = line.rfind("\"");
                    userInput = line.substr(start, end - start);
                    if (userInput.size() > 40) {
                        userInput = userInput.substr(0, 40) + "..."; // Truncate to 40 characters
                    }
                    inConversation = true; // Mark as in a conversation

                    // Get current time (HH:MM)
                    std::time_t now = std::time(nullptr);
                    std::tm* localTime = std::localtime(&now);
                    char timeBuffer[6]; // HH:MM format
                    std::strftime(timeBuffer, sizeof(timeBuffer), "%H:%M", localTime);

                    std::cerr << conversationCount + 1 << ": \"" << conversationTitle << "\" " << timeBuffer << "\n";
                    std::cerr << "\"" << userInput << "\"\n";

                    // Calculate time-sensitive weight
                    double baseWeight = 1.0;
                    double timeFactor = 0.1; // Decay steepness
                    int lastSeenTime = lastSeen[conversationTitle]; // Default 0 if not found
                    int timeDifference = conversationCount - lastSeenTime;
                    double timeWeight = baseWeight / (1.0 + std::exp(-(timeFactor * timeDifference)));

                    // Add neuron with calculated weight
                    addNeuron(timeWeight, userInput);

                    // Update last seen index
                    lastSeen[conversationTitle] = conversationCount;
                    conversationCount++;
                }

                if (line.find("}") != std::string::npos && inConversation) {
                    inConversation = false; // End of a conversation
                }

                // Existing logic for id, content, and parent
                if (line.find("\"id\"") != std::string::npos) {
                    size_t pos = line.find(":") + 1;
                    std::string idStr = line.substr(pos, line.find(",", pos) - pos);
                    idStr.erase(std::remove_if(idStr.begin(), idStr.end(), ::isspace), idStr.end());

                    if (idStr.front() == '"' && idStr.back() == '"') {
                        idStr = idStr.substr(1, idStr.size() - 2);
                    }

                    id = std::stoi(idStr);
                } else if (line.find("\"content\"") != std::string::npos) {
                    size_t start = line.find(":") + 2;
                    size_t end = line.rfind("\"");
                    content = line.substr(start, end - start);
                } else if (line.find("\"parent\"") != std::string::npos) {
                    size_t pos = line.find(":") + 1;
                    std::string parentStr = line.substr(pos, line.find(",", pos) - pos);
                    parentStr.erase(std::remove_if(parentStr.begin(), parentStr.end(), ::isspace), parentStr.end());

                    if (parentStr.front() == '"' && parentStr.back() == '"') {
                        parentStr = parentStr.substr(1, parentStr.size() - 2);
                    }

                    parent = (parentStr == "null") ? -1 : std::stoi(parentStr);
                }

                if (line.find("}") != std::string::npos) {
                    addNeuron(1.0, content);
                    if (parent != -1) {
                        addSynapse(parent, id, 0.5, 1.0, 1.0);
                    }
                    content.clear();
                    parent = -1;
                }
            } catch (const std::exception& e) {
                // Locate the error position within the line
                size_t errorPos = line.find_last_of(":,"); // Simplistic assumption for JSON syntax
                size_t start = (errorPos > 40) ? errorPos - 40 : 0;
                size_t end = std::min(errorPos + 40, line.size());

                // Extract the context around the error
                std::string context = line.substr(start, end - start);

                // Log the exception and context
                std::cerr << "Exception caught: " << e.what() << "\n";
                std::cerr << "Context around error: \"" << context << "\"\n";
            }
        }

        std::cerr << "Finished processing " << conversationCount << " conversations.\n";
        file.close();
    }


    // Relational Coherence: Measures average connectivity density
    double computeRelationalCoherence() const {
        double totalLinks = synapses.size();
        double potentialLinks = neurons.size() * (neurons.size() - 1);
        return potentialLinks > 0 ? totalLinks / potentialLinks : 0.0;
    }

    // Logical Consistency: Checks for contradictory neuron states
    bool checkLogicalConsistency() const {
        for (const auto& neuron : neurons) {
            if (neuron.state > neuron.threshold && neuron.content == "Contradiction") {
                return false;
            }
        }
        return true;
    }

    // Dynamic Adaptability: Tracks weight distribution
    double computeWeightVariance() const {
        double meanWeight = 0.0;
        for (const auto& syn : synapses) {
            meanWeight += syn.weight;
        }
        meanWeight /= synapses.size();

        double variance = 0.0;
        for (const auto& syn : synapses) {
            variance += std::pow(syn.weight - meanWeight, 2);
        }
        return variance / synapses.size();
    }

    // Big picture reporting for humans
    void reportMetrics(int step) const {
        std::cout << "Time Step: " << step << "\n";
        std::cout << "Relational Coherence: " << computeRelationalCoherence() << "\n";
        std::cout << "Logical Consistency: "
                  << (checkLogicalConsistency() ? "Consistent" : "Inconsistent") << "\n";
        std::cout << "Weight Variance: " << computeWeightVariance() << "\n";
        std::cout << "---------------------------------------------\n";
    }

public:
    void addNeuron(double threshold, const std::string& content) {
        neurons.emplace_back(threshold, content);
    }

    void addSynapse(int from, int to, double weight, double delay, double distance) {
        synapses.emplace_back(from, to, weight, delay, distance);
    }

    void forward(double timeStep, int iteration) {
        std::vector<double> newStates(neurons.size(), 0.0);

        for (const auto& syn : synapses) {
            if (neurons[syn.from].state > neurons[syn.from].threshold) {
                double influence = syn.weight * neurons[syn.from].state;
                double timeEffect = std::exp(-timeStep / syn.delay);
                double distanceEffect = std::exp(-syn.distance);

                newStates[syn.to] += influence * timeEffect * distanceEffect;
            }
        }

        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].state = sigmoid(newStates[i]);
        }

        reportMetrics(iteration);
    }

    void train(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << "\n";
            return;
        }

        std::string line, userInput, assistantOutput, feedback;
        int conversationCount = 0;

        while (std::getline(file, line)) {
            try {
                // Parse user input
                if (line.find("\"user_input\"") != std::string::npos) {
                    size_t start = line.find(":") + 2;
                    size_t end = line.rfind("\"");
                    userInput = line.substr(start, end - start);
                }

                // Parse assistant output
                if (line.find("\"assistant_output\"") != std::string::npos) {
                    size_t start = line.find(":") + 2;
                    size_t end = line.rfind("\"");
                    assistantOutput = line.substr(start, end - start);
                }

                // Parse feedback if present
                if (line.find("\"user_feedback\"") != std::string::npos) {
                    size_t start = line.find(":") + 2;
                    size_t end = line.rfind("\"");
                    feedback = line.substr(start, end - start);
                }

                // Process conversation if end detected
                if (line.find("}") != std::string::npos) {
                    if (!userInput.empty() && !assistantOutput.empty()) {
                        // Train network on user input and assistant output
                        processConversation({userInput, assistantOutput, feedback});
                        conversationCount++;
                    }

                    // Clear data for the next conversation
                    userInput.clear();
                    assistantOutput.clear();
                    feedback.clear();
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception caught: " << e.what() << "\n";
            }
        }

        std::cerr << "Training completed: " << conversationCount << " conversations processed.\n";
        file.close();
    }

};

int main(int argc, char* argv[]) {
    Synaptic3RNet net;

    // Determine filepath: default or command-line argument
    std::string filepath = (argc > 1) ? argv[1] : "data/gpt.json";

    // Train the network with the specified file
    std::cout << "Training network with file: " << filepath << "\n";
    net.train(filepath);

    // Continuous operation with big-picture reporting
    int iteration = 0;
    while (true) {
        std::string input;
        std::cout << "Enter input (or type 'exit' to stop): ";
        std::getline(std::cin, input);

        if (input == "exit") {
            std::cout << "Stopping system.\n";
            break;
        }

        std::cout << "Processing input: " << input << "\n";
        net.forward(1.0, iteration++);
    }

    return 0;
}
