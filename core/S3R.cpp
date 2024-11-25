
#include <iostream>
#include <cmath>
#include "S3R.h"

namespace S3R {

// Constructor: Initialize synapse with weights, distance, and force
S3R::S3R(double W, double D, double F)
    : W(W), D(D), F(F) {}

// Update synaptic attributes (weights, distance, force)
void S3R::upd(int it, const S3R& srcN, const S3R& tgtN, double feedback) {
    double dT = std::exp(-0.01 * it); // Time decay
    double dD = 1.0 / (1.0 + std::abs(srcN.W - tgtN.W)); // Distance effect
    double dF = (srcN.F + tgtN.F) / 2.0; // Force average

    // Incorporate feedback adjustment
    double fbAdj = (feedback > 0.0) ? feedback : 1.0;

    // Update attributes
    W *= dT * dD * fbAdj; // Weight adjusted by decay, distance, and feedback
    D = dT * dD * W;      // Distance follows weight adjustments
    F = dF * W;           // Force scales with weight

    // Clamp values to [0, 1]
    W = std::max(0.0, std::min(1.0, W));
    D = std::max(0.0, std::min(1.0, D));
    F = std::max(0.0, std::min(1.0, F));
}

// Evaluate the synapse importance as a combined score
double S3R::eval() const {
    return W * D * F; // Importance is the product of weight, distance, and force
}

// Debugging utility: Output synapse attributes
void S3R::dbg() const {
    std::cout << "S3R [W: " << W << ", D: " << D << ", F: " << F << "]
";
}

} // namespace S3R
