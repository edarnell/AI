#ifndef GPT_H
#define GPT_H

namespace GPT {
    void loadJSON(const std::string& path);
    void iTrain(const std::string& input, const std::string& output);
    void train(unsigned long epochs);
    void init(const std::string& path);
    void save(const std::string& path);
    void load(const std::string& path);
    std::string generateResponse(const std::string& prompt);
    void interactivePrompt();
}

#endif // GPT_H
