#include "wiki.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>

// Helper function to tokenize a string
std::vector<std::string> WikiParser::tokenize(const std::string& text, const std::string& delimiters) {
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;

    while ((end = text.find_first_of(delimiters, start)) != std::string::npos) {
        if (end != start) {
            tokens.push_back(text.substr(start, end - start));
        }
        start = end + 1;
    }

    if (start < text.size()) {
        tokens.push_back(text.substr(start));
    }

    return tokens;
}

// Process a raw article
void WikiParser::processArticle(const std::string& rawArticle) {
    std::regex titleRegex("<title>(.*?)</title>");
    std::regex textRegex("<text.*?>(.*?)</text>");
    std::regex linkRegex("\\[\\[(.*?)\\]\\]");

    std::smatch match;

    // Extract title
    std::string title;
    if (std::regex_search(rawArticle, match, titleRegex)) {
        title = match[1].str();
    }

    // Extract content
    std::string content;
    if (std::regex_search(rawArticle, match, textRegex)) {
        content = match[1].str();
    }

    // Extract links
    std::vector<std::string> links;
    auto linkStart = std::sregex_iterator(rawArticle.begin(), rawArticle.end(), linkRegex);
    auto linkEnd = std::sregex_iterator();
    for (std::sregex_iterator i = linkStart; i != linkEnd; ++i) {
        links.push_back((*i)[1].str());
    }

    // Add article to the map
    articles[title] = {title, content, links};
}

// Load and parse the XML file
void WikiParser::load(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    std::string data = buffer.str();
    std::regex articleRegex("<page>(.*?)</page>");
    auto pageStart = std::sregex_iterator(data.begin(), data.end(), articleRegex);
    auto pageEnd = std::sregex_iterator();

    for (std::sregex_iterator i = pageStart; i != pageEnd; ++i) {
        processArticle((*i)[1].str());
    }

    std::cout << "Loaded " << articles.size() << " articles from " << filePath << std::endl;
}

// Parse a buffer into articles incrementally
std::vector<WikiArticle> WikiParser::parseBuffer(std::string& buffer) {
    std::vector<WikiArticle> articles;

    // Track accumulated data across chunks
    static std::string accumulatedBuffer;

    // Append the current buffer to the accumulated buffer
    accumulatedBuffer += buffer;

    // Find and process complete <page> blocks
    std::size_t start = 0;
    while ((start = accumulatedBuffer.find("<page>", start)) != std::string::npos) {
        std::size_t end = accumulatedBuffer.find("</page>", start);
        if (end == std::string::npos) {
            // No closing tag, wait for more data
            break;
        }

        // Extract the complete <page> block
        std::string pageContent = accumulatedBuffer.substr(start, end - start + 7); // Include "</page>"
        start = end + 7; // Move past the closing tag

        // Parse the <title>, <text>, and links from the <page>
        WikiArticle article;
        std::regex titleRegex(R"(<title>(.*?)</title>)");
        std::regex textRegex(R"(<text.*?>(.*?)</text>)");

        // Extract title
        std::smatch match;
        if (std::regex_search(pageContent, match, titleRegex)) {
            article.title = match[1].str();
        }

        // Extract text
        if (std::regex_search(pageContent, match, textRegex)) {
            article.content = match[1].str();
        }

        // Extract links (example: [[link]])
        std::regex linkRegex(R"(\[\[(.*?)\]\])");
        auto linksBegin = std::sregex_iterator(pageContent.begin(), pageContent.end(), linkRegex);
        auto linksEnd = std::sregex_iterator();
        for (auto it = linksBegin; it != linksEnd; ++it) {
            article.links.push_back((*it)[1].str());
        }

        // Add the article to the list
        articles.push_back(article);
    }

    // Remove processed content from the accumulated buffer
    if (start > 0) {
        accumulatedBuffer.erase(0, start);
    }

    return articles;
}


// Retrieve an article by title
WikiArticle WikiParser::getArticle(const std::string& title) const {
    if (articles.find(title) != articles.end()) {
        return articles.at(title);
    }
    throw std::runtime_error("Article not found: " + title);
}

// Retrieve all article titles
std::vector<std::string> WikiParser::getAllTitles() const {
    std::vector<std::string> titles;
    for (const auto& pair : articles) {
        titles.push_back(pair.first);
    }
    return titles;
}

// Retrieve all articles
std::vector<WikiArticle> WikiParser::getArticles() const {
    std::vector<WikiArticle> articleList;
    for (const auto& pair : articles) {
        articleList.push_back(pair.second);
    }
    return articleList;
}

void WikiParser::extractMetadata(const std::string& buffer) {
    std::size_t start = buffer.find("<siteinfo>");
    std::size_t end = buffer.find("</siteinfo>");
    if (start != std::string::npos && end != std::string::npos) {
        std::string metadataContent = buffer.substr(start, end - start + 11); // Include closing tag

        // Extract key-value pairs using regex
        std::regex fieldRegex(R"(<([^>]+)>([^<]+)</\1>)");
        auto fieldsBegin = std::sregex_iterator(metadataContent.begin(), metadataContent.end(), fieldRegex);
        auto fieldsEnd = std::sregex_iterator();

        for (auto it = fieldsBegin; it != fieldsEnd; ++it) {
            std::string key = (*it)[1].str();
            std::string value = (*it)[2].str();
            metadata[key] = value;
        }

        // Print extracted metadata for debugging
        std::cout << "Extracted Metadata:\n";
        for (const auto& [key, value] : metadata) {
            std::cout << " - " << key << ": " << value << "\n";
        }
    } else {
        std::cerr << "No <siteinfo> metadata found in the buffer.\n";
    }
}

std::vector<float> WikiParser::extractFeatures(const WikiArticle& article) {
    std::vector<float> features;

    // Add length of the content as a feature
    features.push_back(static_cast<float>(article.content.size()));

    // Add the number of links as a feature
    features.push_back(static_cast<float>(article.links.size()));

    // Add metadata-derived features (e.g., number of keys)
    features.push_back(static_cast<float>(metadata.size()));

    return features;
}


