
#include "L3R.h"
#include "N3R.h"
#include "S3R.h"
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace N3R {

// Constructor
L3R::L3R() : errorCount(0) {}

// Infer logical operators dynamically, including nested logic
bool L3R::isPotentialOperator(const std::string& token, const std::string& prevToken, const std::string& nextToken) const {
    static const std::unordered_map<std::string, std::string> logicalSynonyms = {
        {"AND", "AND"}, {"&&", "AND"}, {"with", "AND"}, {"plus", "AND"},
        {"OR", "OR"}, {"||", "OR"}, {"either", "OR"}, {"or", "OR"},
        {"NOT", "NOT"}, {"!", "NOT"}, {"never", "NOT"},
        {"IMPLIES", "IMPLIES"}, {"->", "IMPLIES"}, {"leads to", "IMPLIES"}
    };

    // Direct match
    if (logicalSynonyms.find(token) != logicalSynonyms.end()) return true;

    // Infer from sentence structure
    if (prevToken == "if" && token == "then") return true; // Implies
    if (token == "," && nextToken == "or") return true; // Logical OR

    // Highlight ambiguity dynamically
    if (!token.empty()) {
        std::cout << "Ambiguous token: " << token
                  << ". Could this be a logical operator or operand? Please clarify.
";
    }
    return false;
}

// Validate logical input as a pure logical statement
bool L3R::validateLogicalExpression(const std::string& input) {
    std::istringstream iss(input);
    std::string token, prevToken, nextToken;
    int operatorCount = 0, operandCount = 0;
    std::vector<std::string> tokens;

    // Tokenize input
    while (iss >> token) {
        tokens.push_back(token);
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        prevToken = (i > 0) ? tokens[i - 1] : "";
        token = tokens[i];
        nextToken = (i < tokens.size() - 1) ? tokens[i + 1] : "";

        if (isPotentialOperator(token, prevToken, nextToken)) {
            operatorCount++;
        } else {
            operandCount++;
        }
    }

    if (operatorCount >= operandCount) {
        errorCount++;
        failedTests.push_back("Invalid logical expression: " + input);
        return false;
    }
    return true;
}

// Detect ambiguities and provide structured metadata
AmbiguityInfo L3R::detectAmbiguity(const std::string& input, const std::string& context) const {
    AmbiguityInfo info;
    info.context = context;

    // Tokenize input and analyze
    std::istringstream iss(input);
    std::string token, prevToken, nextToken;
    std::vector<std::string> tokens;
    while (iss >> token) tokens.push_back(token);

    for (size_t i = 0; i < tokens.size(); ++i) {
        prevToken = (i > 0) ? tokens[i - 1] : "";
        token = tokens[i];
        nextToken = (i < tokens.size() - 1) ? tokens[i + 1] : "";

        if (!isPotentialOperator(token, prevToken, nextToken)) {
            info.phrase = token;
            info.interpretations = {"Operator?", "Operand?"}; // Example options
            info.confidence = 0.5; // Low confidence due to ambiguity
            break;
        }
    }
    return info;
}

// Debugging utility: Extend to include logic tests and modeling states
void L3R::dbg() const {
    std::cout << "Logical State: Error count = " << errorCount << "
";

    // Debug logical validation results
    for (const auto& test : failedTests) {
        std::cout << "Failed Test: " << test << "
";
    }

    // Debug NNet states
    if (NNet) {
        NNet->dbg();
    }

    // Debug S3R synapses
    for (const auto& syn : synapses) {
        syn.dbg();
    }
}

} // namespace N3R
