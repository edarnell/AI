
#ifndef S3R_H
#define S3R_H

namespace S3R {

class S3R {
public:
    double w; // Synaptic weight
    double d; // Synaptic distance
    double f; // Synaptic force

    // Constructor: Initialize synapse with weights, distance, and force
    S3R(double w = 0.0, double d = 0.0, double f = 0.0);

    // Update synaptic attributes (weights, distance, force)
    void upd(int n, const S3R& s, const S3R& t, double feedback);

    // Evaluate the synapse importance as a combined score
    double eval() const;

    // Debugging utility: Output synapse attributes
    void dbg() const;
};

} // namespace S3R

#endif // S3R_H
