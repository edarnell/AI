#include "wiki.h"
#include "neural.h"
#include <iostream>
#include <fstream>
#include <bzlib.h>

void processChunk(WikiParser& parser, NeuralNet*& net, std::string& articleBuffer, int& articleCount, int& totalLinks, int& totalWords) {
    // Process metadata if present
    if (articleBuffer.find("<siteinfo>") != std::string::npos) {
        parser.extractMetadata(articleBuffer);
    }

    // Process articles from the buffer
    auto articles = parser.parseBuffer(articleBuffer);
    std::cout << "Articles parsed: " << articles.size() << "\n";

    for (const auto& article : articles) {
        // Preprocess the article content using NeuralNet
        auto input = net->preprocessArticle(article.content);

        // Extract features from the article
        auto features = parser.extractFeatures(article); // Pass `article`, not `article.content`

        // Combine input and features
        input.insert(input.end(), features.begin(), features.end());

        // Adjust NeuralNet dynamically if input size changes
        if (!net || net->getInputNeuronCount() != input.size()) {
            if (net) delete net; // Free old instance
            net = new NeuralNet(std::vector<int>{(int)input.size(), 64, 10}); // Example hidden layer sizes
            std::cout << "NeuralNet reinitialized with input size: " << input.size() << "\n";
        }

        // Forward propagate
        try {
            net->forwardPropagate(input);
        } catch (const std::exception& e) {
            std::cerr << "Error during forward propagation: " << e.what() << "\n";
            continue; // Skip problematic article
        }

        // Track statistics
        totalLinks += article.links.size();
        totalWords += article.content.size();
        ++articleCount;

        // Debugging for features
        std::cout << "Processed article. Features extracted: " << features.size()
                  << ", Links: " << article.links.size()
                  << ", Words: " << article.content.size() << "\n";
    }

    std::cout << "Processed articles: " << articleCount
              << ", Total links: " << totalLinks
              << ", Total words: " << totalWords << "\n";

    articleBuffer.clear(); // Reset buffer
}

int main() {
    try {
        // Initialize WikiParser
        WikiParser parser;

        // Open the compressed Wikipedia dump
        std::string filePath = "data/wiki.xml.bz2";
        FILE* file = fopen(filePath.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        int bzError;
        BZFILE* bzf = BZ2_bzReadOpen(&bzError, file, 0, 0, nullptr, 0);
        if (bzError != BZ_OK) {
            fclose(file);
            throw std::runtime_error("Error during bz2 read operation. bzError code: " + std::to_string(bzError));
        }

        char buffer[4096];
        std::string articleBuffer;

        // Track progress
        int articleCount = 0;
        int totalLinks = 0;
        int totalWords = 0;

        NeuralNet* net = nullptr; // Pointer for dynamic initialization

        // Read and process chunks of the file
        while (true) {
            int bytesRead = BZ2_bzRead(&bzError, bzf, buffer, sizeof(buffer));
            if (bytesRead > 0) {
                articleBuffer.append(buffer, bytesRead);

                // Process the buffer if it contains at least one complete article
                if (articleBuffer.find("</page>") != std::string::npos) {
                    std::cout << "Processing chunk. Buffer size: " << articleBuffer.size() << "\n";
                    processChunk(parser, net, articleBuffer, articleCount, totalLinks, totalWords);
                }
            } else if (bzError == BZ_STREAM_END) {
                std::cout << "End of compressed stream reached.\n";
                break;
            } else {
                throw std::runtime_error("Error during bz2 read operation. bzError code: " + std::to_string(bzError));
            }
        }

        // Clean up
        if (net) {
            delete net; // Free dynamically allocated NeuralNet
        }
        BZ2_bzReadClose(&bzError, bzf);
        fclose(file);
        std::cout << "Processing complete. Total articles: " << articleCount << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
