#ifndef N3R_H
#define N3R_H

#include <string>
#include <unordered_map>
#include <vector>

namespace N3R {

// Node structure
struct N {
    std::string id; // Node identifier
    double W;       // Weight
};

// Synapse structure
struct S {
    std::string src; // Source node
    std::string tgt; // Target node
    double W;        // Weight
    double D;        // Delay
};

// Neural Network class
class NNet {
public:
    NNet();                              // Constructor
    void addN(const std::string& id, double W);       // Add node
    void addS(const std::string& src, const std::string& tgt, double W, double D); // Add synapse
    void updW(int it);                  // Update weights
    void fwd(double I, int it);         // Forward propagation
    void lowW() const;                  // Expose low-confidence nodes and synapses

private:
    std::unordered_map<std::string, N> N; // Nodes
    std::vector<S> S;                     // Synapses
};

} // namespace N3R

#endif // N3R_H
