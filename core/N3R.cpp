
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include "N3R.h"

namespace N3R {

// Constructor
NNet::NNet() {
    std::cout << "Initializing Neural Network...
";
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

// Update synaptic weights with decay, distance, and feedback
void NNet::uS(int t, const std::vector<double>& fb) {
    size_t iF = 0;
    for (auto& s : ss) {
        if (ns.find(s.s) != ns.end() && ns.find(s.t) != ns.end()) {
            double dT = std::exp(-0.01 * t); // Time decay
            double dD = 1.0 / (1.0 + std::abs(ns[s.s].w - ns[s.t].w)); // Distance effect
            double adjFb = (iF < fb.size()) ? fb[iF++] : 1.0; // Feedback adjustment

            s.w *= dT;  // Apply time decay
            s.w *= dD;  // Apply distance effect
            s.w *= adjFb; // Apply feedback adjustment
            s.w = std::max(0.0, std::min(1.0, s.w)); // Clamp to [0, 1]
        }
    }
}

// Forward propagate with input, time decay, and feedback
void NNet::fwd(double I, int t, const std::vector<double>& fb) {
    double dT = std::exp(-0.01 * t); // Time decay
    size_t iF = 0;

    for (auto& [id, n] : ns) {
        double dD = 1.0 / (1.0 + n.w); // Distance effect
        double adjFb = (iF < fb.size()) ? fb[iF++] : 1.0; // Feedback adjustment

        double adj = I * dT * dD * adjFb; // Combine adjustments
        n.w += adj;
        n.w = std::max(0.0, std::min(1.0, n.w)); // Clamp to [0, 1]
    }

    // Update synaptic weights
    uS(t, fb);
}

// Expose nodes and synapses with low confidence (weight < 0.5)
void NNet::lowW() const {
    for (const auto& [id, n] : ns) {
        if (n.w < 0.5) {
            std::cout << "Low-confidence node: " << id << " with weight " << n.w << "
";
        }
    }
    for (const auto& s : ss) {
        if (s.w < 0.5) {
            std::cout << "Low-confidence synapse: " << s.s << " -> " << s.t
                      << " with weight " << s.w << "
";
        }
    }
}

// Debugging Utility
void NNet::dbg() const {
    std::cout << "Node States:
";
    for (const auto& [id, n] : ns) {
        std::cout << "Node: " << id << " Weight: " << n.w << "
";
    }
    std::cout << "Synapse States:
";
    for (const auto& s : ss) {
        std::cout << "Synapse: " << s.s << " -> " << s.t
                  << " Weight: " << s.w << " Distance: " << s.d << "
";
    }
}

} // namespace N3R
