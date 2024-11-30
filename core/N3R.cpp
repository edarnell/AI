#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include "N3R.h"

namespace N3R {

// Constructor
NNet::NNet() {
    std::cout << "Initializing Neural Network...\n";
}

// Add Node
void NNet::addN(const std::string& n, double w) {
    if (ns.find(n) == ns.end()) {
        ns[n] = {n, w};
    }
}

// Add Synapse
void NNet::addS(const std::string& s, const std::string& t, double w, double d) {
    ss.push_back({s, t, w, d});
}

// Update nodes with external feedback
void NNet::uN(const std::vector<double>& fb) {
    size_t iF = 0;
    for (auto& [id, n] : ns) {
        double adjFb = (iF < fb.size()) ? fb[iF++] : 0.0; // Feedback adjustment
        n.w += adjFb;
        n.w = std::max(0.0, std::min(1.0, n.w)); // Clamp to [0, 1]
    }
}

// Forward propagate with input, time decay, and feedback
void NNet::fwd(double I, int n) {
    double dT = std::exp(-0.01 * n); // Time decay
    size_t iF = 0;

    for (auto& [id, n] : ns) {
        double dD = 1.0 / (1.0 + n.w); // Distance effect
        double adjFb = (iF < fb.size()) ? fb[iF++] : 1.0; // Feedback adjustment

        double adj = I * dT * dD * adjFb; // Combine adjustments
        n.w += adj;
        n.w = std::max(0.0, std::min(1.0, n.w)); // Clamp to [0, 1]
    }

    // Update synaptic weights
    uS(n, fb);
}

// Expose nodes and synapses with low confidence (weight < 0.5)
void NNet::lowW() const {
    std::cout << "Low-confidence Nodes and Synapses:\n";
    for (const auto& [id, n] : ns) {
        if (n.w < 0.5) {
            std::cout << "  Node [" << id << "] Weight: " << n.w << "\n";
        }
    }
    for (const auto& s : ss) {
        if (s.w < 0.5) {
            std::cout << "  Synapse [" << s.s << " -> " << s.t
                      << "] Weight: " << s.w << "\n";
        }
    }
}

// Debugging Utility
void NNet::dbg() const {
    std::cout << "Node States:\n";
    for (const auto& [id, n] : ns) {
        std::cout << "  Node [" << id << "] Weight: " << n.w << "\n";
    }
    std::cout << "Synapse States:\n";
    for (const auto& s : ss) {
        std::cout << "  Synapse [" << s.s << " -> " << s.t
                  << "] Weight: " << s.w << " Distance: " << s.d << "\n";
    }
}

} // namespace N3R

