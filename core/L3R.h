
#ifndef L3R_H
#define L3R_H

#include <string>
#include <vector>
#include <unordered_map>

namespace N3R {
// forward defs
class NNet;
class S3R;

// Structure for ambiguity metadata
struct AmbiguityInfo {
    std::string phrase;                        // Ambiguous phrase or token
    std::string context;                       // Context detected (e.g., system logic, physics)
    std::vector<std::string> interpretations; // Possible meanings
    double confidence;                         // Confidence level of the primary interpretation
};
class L3R {
public:
    // Constructor
    L3R();

    // Logical validation
    bool validateLogicalExpression(const std::string& input);
    bool validateInput(const std::string& input);

    // Ambiguity detection
    AmbiguityInfo detectAmbiguity(const std::string& input, const std::string& context) const;

    // Operator inference
    bool isPotentialOperator(const std::string& token, const std::string& prevToken, const std::string& nextToken) const;

    // Debugging utility
    void dbg() const;

private:
    size_t errorCount;                         // Count of detected errors
    std::vector<std::string> failedTests;      // List of failed tests for debugging

    // Reference to network and synapse models
    NNet* net;                                 // Pointer to the N3R network
    std::vector<S3R> synapses;                 // Vector of S3R synapses
};

} // namespace N3R

#endif // L3R_H
