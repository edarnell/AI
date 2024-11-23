#ifndef WIKI_H
#define WIKI_H

#include <string>
#include <vector>
#include <unordered_map>

// Struct for representing a Wikipedia article
struct WikiArticle {
    std::string title;
    std::string content;
    std::vector<std::string> links; // Outgoing links to other articles
};

// Class for parsing and handling Wikipedia data
class WikiParser {
private:
    std::unordered_map<std::string, WikiArticle> articles; // Map of articles by title
    std::vector<std::string> tokenize(const std::string& text, const std::string& delimiters);
    void processArticle(const std::string& rawArticle);

public:
    WikiParser() = default;
void extractMetadata(const std::string& buffer);
    void load(const std::string& filePath); // Load and parse the XML file
    std::vector<WikiArticle> parseBuffer(std::string& buffer); // Incremental buffer parser
    WikiArticle getArticle(const std::string& title) const; // Retrieve an article by title
    std::vector<std::string> getAllTitles() const; // Retrieve all article titles
    std::vector<WikiArticle> getArticles() const; // Retrieve all articles
    std::vector<float> extractFeatures(const WikiArticle& article);
    std::unordered_map<std::string, std::string> metadata;
};

#endif
