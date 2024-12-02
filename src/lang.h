#ifndef LANG_H
#define LANG_H

#include <string>
#include <vector>
#include <unordered_map>

namespace Lang {

// Tokenizes input text into words
std::vector<std::string> tokenize(const std::string& text);

// Calculates Jaccard similarity between two token sets
double calcJdSim(const std::vector<std::string>& a, const std::vector<std::string>& b);

// Parses structured feedback and returns categorized data
std::unordered_map<std::string, std::string> parseFeedback(const std::string& feedback);

// Analyzes feedback for error patterns
std::unordered_map<std::string, int> analyzePatterns(const std::string& feedback);

// Evaluates text similarity for potential training improvements
bool evaluateSimilarity(const std::string& input, const std::string& context, double threshold);

} // namespace Lang

#endif // LANG_H
