
#include "L3R.h"
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace N3R {

// Constructor
L3R::L3R() : errorCount(0) {}

// Dynamic inference of logical operators based on context
bool L3R::isPotentialOperator(const std::string& token, const std::string& prevToken, const std::string& nextToken) const {
    // Common patterns for logical operators
    static const std::vector<std::string> negationCues = {"not", "without", "never"};
    static const std::vector<std::string> conjunctionCues = {"and", ",", "with"};
    static const std::vector<std::string> disjunctionCues = {"or", "either"};

    // Check for negation
    if (std::find(negationCues.begin(), negationCues.end(), token) != negationCues.end() ||
        token == "!" || (prevToken == "no" && nextToken == "")) {
        return true; // Implying NOT
    }

    // Check for conjunction
    if (std::find(conjunctionCues.begin(), conjunctionCues.end(), token) != conjunctionCues.end()) {
        return true; // Implying AND
    }

    // Check for disjunction
    if (std::find(disjunctionCues.begin(), disjunctionCues.end(), token) != disjunctionCues.end()) {
        return true; // Implying OR
    }

    return false;
}

// Evaluate realism dynamically based on inferred operators and context
double L3R::evaluateRealism(const std::string& input) {
    if (input.empty()) {
        errorCount++;
        failedTests.push_back("Empty input provided for realism evaluation");
        std::cout << "Ambiguity: Input is empty. Cannot evaluate realism.
";
        return 0.0; // Completely unrealistic
    }

    double realismScore = 1.0; // Start fully realistic
    std::istringstream iss(input);
    std::string token, prevToken, nextToken;
    std::vector<std::string> tokens;

    while (iss >> token) {
        tokens.push_back(token);
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        prevToken = (i > 0) ? tokens[i - 1] : "";
        nextToken = (i < tokens.size() - 1) ? tokens[i + 1] : "";

        if (isPotentialOperator(tokens[i], prevToken, nextToken)) {
            realismScore *= dynamicAdjustOperator(tokens[i], prevToken, nextToken);
        }
    }

    realismScore = std::max(0.0, std::min(1.0, realismScore)); // Clamp to [0, 1]
    std::cout << "Realism score: " << realismScore << "
";
    return realismScore;
}

// Dynamic adjustment based on operator and context
double L3R::dynamicAdjustOperator(const std::string& token, const std::string& prevToken, const std::string& nextToken) const {
    double adjustment = 1.0;

    if (token == "and" || token == ",") {
        adjustment -= 0.05; // Conjunctions reduce certainty slightly
    } else if (token == "or" || token == "either") {
        adjustment -= 0.1; // Disjunctions increase uncertainty
    } else if (token == "not" || token == "!") {
        adjustment -= 0.2; // Negations reduce certainty further
    } else if (prevToken == "no" && nextToken.empty()) {
        adjustment -= 0.15; // Implicit negation from "no"
    }

    std::cout << "Operator "" << token << "" adjusted realism by: " << adjustment << "
";
    return adjustment;
}

// Debug logical state
void L3R::dbg() const {
    std::cout << "Logical State: Error count = " << errorCount << "
";
}

} // namespace N3R
