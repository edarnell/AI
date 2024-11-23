#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stack>

// Data structures for parsed conversations
struct Message {
    std::string id;
    std::string parent;
    std::string author;
    std::string content;
};

struct Conversation {
    std::string title;
    std::string createTime;
    std::string updateTime;
    std::vector<Message> messages;
};

// JSON Parser Class
class ChatParser {
private:
    std::vector<Conversation> conversations;

    // Utility to strip quotes from a string
    std::string stripQuotes(const std::string &str) {
        if (str.size() >= 2 && str[0] == '"' && str.back() == '"') {
            return str.substr(1, str.size() - 2);
        }
        return str;
    }

public:
    // Parse JSON from chat.json file
    void parseJSON(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        std::string line, currentTitle, currentCreateTime, currentUpdateTime;
        Conversation conversation;
        Message message;
        bool inMessages = false;

        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t")); // Trim leading whitespace

            // Parse title
            if (line.find("\"title\":") != std::string::npos) {
                if (!conversation.title.empty()) {
                    conversations.push_back(conversation);
                    conversation = Conversation(); // Reset for next conversation
                }
                currentTitle = stripQuotes(line.substr(line.find(":") + 1));
                conversation.title = currentTitle;
            }

            // Parse create_time
            if (line.find("\"create_time\":") != std::string::npos) {
                currentCreateTime = stripQuotes(line.substr(line.find(":") + 1));
                conversation.createTime = currentCreateTime;
            }

            // Parse update_time
            if (line.find("\"update_time\":") != std::string::npos) {
                currentUpdateTime = stripQuotes(line.substr(line.find(":") + 1));
                conversation.updateTime = currentUpdateTime;
            }

            // Enter messages block
            if (line.find("\"mapping\":") != std::string::npos) {
                inMessages = true;
                continue;
            }

            // Parse message details
            if (inMessages) {
                if (line.find("\"id\":") != std::string::npos) {
                    message.id = stripQuotes(line.substr(line.find(":") + 1));
                } else if (line.find("\"parent\":") != std::string::npos) {
                    message.parent = stripQuotes(line.substr(line.find(":") + 1));
                } else if (line.find("\"role\":") != std::string::npos) {
                    message.author = stripQuotes(line.substr(line.find(":") + 1));
                } else if (line.find("\"parts\":") != std::string::npos) {
                    message.content = stripQuotes(line.substr(line.find("[") + 1));
                    conversation.messages.push_back(message);
                    message = Message(); // Reset for next message
                }

                // Exit messages block
                if (line.find("}") != std::string::npos && !message.id.empty()) {
                    inMessages = false;
                }
            }
        }

        // Add the last conversation
        if (!conversation.title.empty()) {
            conversations.push_back(conversation);
        }
    }

    // Display parsed conversations
    void displayConversations() const {
        for (const auto &conv : conversations) {
            std::cout << "Title: " << conv.title << "\n"
                      << "Created: " << conv.createTime << "\n"
                      << "Updated: " << conv.updateTime << "\n"
                      << "Messages:\n";

            for (const auto &msg : conv.messages) {
                std::cout << "  ID: " << msg.id << "\n"
                          << "  Parent: " << msg.parent << "\n"
                          << "  Author: " << msg.author << "\n"
                          << "  Content: " << msg.content << "\n";
            }
            std::cout << "---------------------------------\n";
        }
    }
};

int main() {
    ChatParser parser;

    // Parse chat.json
    std::string filename = "data/chat.json"; // Replace with actual file path
    parser.parseJSON(filename);

    // Display parsed conversations
    parser.displayConversations();

    return 0;
}
