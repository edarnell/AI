#ifndef N3R_H
#define N3R_H

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace N3R {

enum VM { Node, Synapse, Model }; // Validation Modes: Node, Synapse, or Model

struct N {
    std::string n; // Node ID
    double w;      // Weight
};

class S3R {
public:
    double w;   // Synaptic weight
    double d;   // Synaptic distance
    double f;   // Synaptic force
    std::string s;  // Source node ID
    std::string t;  // Target node ID

    S3R(const std::string& s, const std::string& t, double w = 0.0, double d = 0.0, double f = 0.0);

    void upd(const N& sn, const N& tn, double fb); // Update weights
    std::pair<double, double> eval() const;       // Returns {confidence, uncertainty}
    void dbg() const;                             // Debug synapse state
};

class NNet {
public:
    NNet();

    void addN(const std::string& n, double w); 
    void addS(const std::string& s, const std::string& t, double w, double d); 
    void fwd(double I, int n); 
    void validate(VM mode, double confThr = 0.05, double lowXThr = 0.2, double weightThr = 0.999, double distThr = 0.001) const; 

private:
    std::unordered_map<std::string, N> ns; // Nodes
    std::vector<S3R> ss;                   // Synapses

    // Validation helpers
    void validateNode(const N& node) const;                             // Validate individual node
    void validateSynapse(const S3R& syn, double confThr, double lowXThr, double weightThr, double distThr) const; // Validate individual synapse
    void validateModel(double confThr, double lowXThr, double weightThr, double distThr) const; // Validate entire model
    double calcThr(const S3R& syn, double confThr) const;               // Calculate dynamic threshold
    bool isConnected() const;                        // Check if the network is fully connected
    void dfs(const std::string& id, std::unordered_map<std::string, bool>& visited) const; // DFS for connectivity
    bool hasCycle() const;                           // Check for cyclic dependencies
    bool hasCycleUtil(const std::string& id, std::unordered_map<std::string, bool>& visited, std::unordered_map<std::string, bool>& stack) const; // Cycle helper
    double calculateAverageWeight() const;           // Compute average synapse weight
};

} // namespace N3R

#endif // N3R_H

