
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
void NNet::addN(const std::string& id, double W) {
    if (N.find(id) == N.end()) {
        N[id] = {id, W};
    }
}

// Add Synapse
void NNet::addS(const std::string& src, const std::string& tgt, double W, double D) {
    S.push_back({src, tgt, W, D});
}

// Update nodes with external feedback
void NNet::uN(const std::vector<double>& fb) {
    size_t iF = 0;
    for (auto& [id, n] : N) {
        double adjFb = (iF < fb.size()) ? fb[iF++] : 0.0; // Feedback adjustment
        n.W += adjFb;
        n.W = std::max(0.0, std::min(1.0, n.W)); // Clamp to [0, 1]
    }
}

// Update synaptic weights with decay, distance, and feedback
void NNet::uS(int t, const std::vector<double>& fb) {
    size_t iF = 0;
    for (auto& s : S) {
        if (N.find(s.src) != N.end() && N.find(s.tgt) != N.end()) {
            double dT = std::exp(-0.01 * t); // Time decay
            double dD = 1.0 / (1.0 + std::abs(N[s.src].W - N[s.tgt].W)); // Distance effect
            double adjFb = (iF < fb.size()) ? fb[iF++] : 1.0; // Feedback adjustment

            s.W *= dT;  // Apply time decay
            s.W *= dD;  // Apply distance effect
            s.W *= adjFb; // Apply feedback adjustment
            s.W = std::max(0.0, std::min(1.0, s.W)); // Clamp to [0, 1]
        }
    }
}

// Forward propagate with input, time decay, and feedback
void NNet::fwd(double I, int t, const std::vector<double>& fb) {
    double dT = std::exp(-0.01 * t); // Time decay
    size_t iF = 0;

    for (auto& [id, n] : N) {
        double dD = 1.0 / (1.0 + n.W); // Distance effect
        double adjFb = (iF < fb.size()) ? fb[iF++] : 1.0; // Feedback adjustment

        double adj = I * dT * dD * adjFb; // Combine adjustments
        n.W += adj;
        n.W = std::max(0.0, std::min(1.0, n.W)); // Clamp to [0, 1]
    }

    // Update synaptic weights
    uS(t, fb);
}

// Expose nodes and synapses with low confidence (weight < 0.5)
void NNet::lowW() const {
    for (const auto& [id, n] : N) {
        if (n.W < 0.5) {
            std::cout << "Low-confidence node: " << id << " with weight " << n.W << "
";
        }
    }
    for (const auto& s : S) {
        if (s.W < 0.5) {
            std::cout << "Low-confidence synapse: " << s.src << " -> " << s.tgt
                      << " with weight " << s.W << "
";
        }
    }
}

// Debugging Utility
void NNet::dbg() const {
    std::cout << "Node States:
";
    for (const auto& [id, n] : N) {
        std::cout << "Node: " << id << " Weight: " << n.W << "
";
    }
    std::cout << "Synapse States:
";
    for (const auto& s : S) {
        std::cout << "Synapse: " << s.src << " -> " << s.tgt
                  << " Weight: " << s.W << " Distance: " << s.D << "
";
    }
}

} // namespace N3R
