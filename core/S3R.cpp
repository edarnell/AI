#include <cmath>
#include "S3R.h"

namespace N3R {

// Constructor
S3R::S3R(double W, double D, double F)
    : W(W), D(D), F(F) {}

// Update Synapse Weights
void S3R::upd(const N& srcN, const N& tgtN, int it) {
    // Dimensional adjustments
    double dT = std::exp(-0.01 * it);  // Time decay
    double dD = 1.0 / (1.0 + std::abs(srcN.W - tgtN.W));  // Relative distance
    double dF = (srcN.F + tgtN.F) / 2.0;  // Force average

    // Apply changes to synaptic weights
    W *= dT;
    D *= dD;
    F *= dF;

    // Clamp values to valid range
    W = std::max(0.0, std::min(1.0, W));
    D = std::max(0.0, std::min(1.0, D));
    F = std::max(0.0, std::min(1.0, F));
}

// Evaluate Synapse Importance
double S3R::eval() const {
    return W * D * F; // Combined importance score
}

// Debug Synapse Information
void S3R::dbg() const {
    std::cout << "Synapse Info: [W=" << W << ", D=" << D << ", F=" << F << "]\n";
}

} // namespace N3R
