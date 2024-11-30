#ifndef N3R_H
#define N3R_H

#include <string>
#include <unordered_map>
#include <vector>
#include <utility> // For std::pair

namespace N3R {

// Node structure
struct N {
    std::string n; // Node identifier
    double w;      // Weight
};

// Synapse class (now part of N3R.h)
class S3R {
public:
    double w;       // Synaptic weight
    double d;       // Synaptic distance
    double f;       // Synaptic force
    std::string s;  // Source node ID
    std::string t;  // Target node ID

    S3R(const std::string& s, const std::string& t, double w = 0.0, double d = 0.0, double f = 0.0);

    void upd(const S3R& sN, const S3R& tN, double feedback);
    std::pair<double, double> eval() const; // Returns {confidence, uncertainty}
    void dbg() const;
};

// Neural Network class
class NNet {
public:
    NNet();                             // Constructor
    void addN(const std::string& n, double w);       // Add node
    void addS(const std::string& s, const std::string& t, double w, double d); // Add synapse
    void uN(const std::vector<double>& fb);          // Update nodes
    void fwd(double I, int n);          // Forward propagation
    void lowW() const;                  // Expose low-confidence nodes and synapses
    void dbg() const;                   // Debugging utility

private:
    std::unordered_map<std::string, N> ns; // Nodes
    std::vector<S3R> ss;                   // Synapses
};

} // namespace N3R

#endif // N3R_H

