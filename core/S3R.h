
#ifndef S3R_H
#define S3R_H

namespace S3R {

class S3R {
public:
    double W; // Synaptic weight
    double D; // Synaptic distance
    double F; // Synaptic force

    // Constructor: Initialize synapse with weights, distance, and force
    S3R(double W = 0.0, double D = 0.0, double F = 0.0);

    // Update synaptic attributes (weights, distance, force)
    void upd(int it, const S3R& srcN, const S3R& tgtN, double feedback);

    // Evaluate the synapse importance as a combined score
    double eval() const;

    // Debugging utility: Output synapse attributes
    void dbg() const;
};

} // namespace S3R

#endif // S3R_H
