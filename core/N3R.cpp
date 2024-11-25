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
void NNet::addN(const std::string& id, double W) {
    if (N.find(id) == N.end()) {
        N[id] = {id, W};
    }
}

// Add Synapse
void NNet::addS(const std::string& src, const std::string& tgt, double W, double D) {
    S.push_back({src, tgt, W, D});
}

// Update Weights
void NNet::updW(int it) {
    for (auto& syn : S) {
        if (N.find(syn.src) != N.end() && N.find(syn.tgt) != N.end()) {
            double dT = std::exp(-0.01 * it); // Time decay
            double dD = 1.0 / (1.0 + std::abs(N[syn.src].W - N[syn.tgt].W)); // Distance effect
            syn.W *= dT * dD;
        }
    }
}

// Forward Propagation
void NNet::fwd(double I, int it) {
    double dT = std::exp(-0.01 * it); // Time decay factor
    for (auto& [id, n] : N) {
        // Relativistic feedback adjustment
        double dD = 1.0 / (1.0 + n.W); // Distance-like effect
        double dF = I * dT;            // Force-like effect

        // Combine dimensions
        double adjustment = dF * dD;
        n.W += adjustment;

        // Clamp weights to [0.0, 1.0]
        n.W = std::max(0.0, std::min(1.0, n.W));
    }

    // Update synaptic weights
    updW(it);
}

// Expose Uncertainty
void NNet::lowW() const {
    for (const auto& [id, n] : N) {
        if (n.W < 0.5) {
            std::cout << "Low-confidence node: " << id << " with weight " << n.W << "\n";
        }
    }
    for (const auto& syn : S) {
        if (syn.W < 0.5) {
            std::cout << "Low-confidence synapse: " << syn.src << " -> " << syn.tgt
                      << " with weight " << syn.W << "\n";
        }
    }
}

} // namespace N3R
