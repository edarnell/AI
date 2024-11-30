#ifndef N3R_H
#define N3R_H

#include <string>
#include <unordered_map>
#include <vector>

namespace N3R {

// Node structure
struct N {
    std::string n; // Node identifier
    double w;       // Weight
};

// Synapse structure
struct S {
    std::string s; // Source node
    std::string t; // Target node
    double w;        // Weight
    double d;        // Delay
};

// Neural Network class
class NNet {
public:
    NNet();                              // Constructor
    void addN(const std::string& n, double w);       // Add node
    void addS(const std::string& s, const std::string& t, double w, double d); // Add synapse
    void updW(int it);                  // Update weights
    void fwd(double I, int it);         // Forward propagation
    void lowW() const;                  // Expose low-confidence nodes and synapses

private:
    std::unordered_map<std::string, N> ns; // Nodes
    std::vector<S> ss;                     // Synapses
};

} // namespace N3R

#endif // N3R_H
