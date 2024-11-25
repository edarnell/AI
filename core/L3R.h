
#ifndef L3R_H
#define L3R_H

#include <string>
#include <vector>
#include <utility>

namespace N3R {

class L3R {
public:
    L3R();                                  // Constructor

    // Core logic checks
    bool validateInput(const std::string& input); // Validate individual input logic
    void relationalCheck(const std::vector<std::pair<std::string, std::string>>& relations); // Relational logic validation
    void consistencyCheck();                      // Perform overall logical consistency check

    // Debugging and test harness
    void dbg() const;                             // Debug logical state
    void runTests();                              // Test harness for logical evaluation

private:
    int errorCount;           // Tracks logical errors
    std::vector<std::string> failedTests; // Stores failed test cases

    // Helper functions
    bool isRelationallyValid(const std::string& a, const std::string& b) const;
};

} // namespace N3R

#endif // L3R_H
