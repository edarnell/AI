#ifndef N3R_H
#define N3R_H

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace N3R {
    // Represents a node in the network.
    struct Node {
        std::string id;  // Unique identifier
        std::string type; // Type: "input", "hidden", or "output"
        float value;     // Current value
    };
    // Represents a synapse (connection) in the network.
    struct Synapse {
        std::string src;  // Source node ID
        std::string dest; // Destination node ID
        float weight;     // Synaptic weight
    };
    //Represents the neural network.
    class NNet {
    private:
        std::unordered_map<std::string, Node> nodes; // Nodes in the network
        
        void validateNodes() const; //Validate the nodes in the network.
        void validateSynapses() const; // Validate the synapses in the network.

        /**
         * @brief Check for cycles in the network using depth-first search.
         * @param nodeId The current node ID being checked.
         * @param visited Map tracking visited nodes.
         * @param stack Map tracking the recursion stack.
         * @return True if a cycle is detected, otherwise false.
         */
        bool dfsCycleCheck(const std::string& nodeId,
                           std::unordered_map<std::string, bool>& visited,
                           std::unordered_map<std::string, bool>& stack) const;
        void checkCycles() const; // Check for cycles in the network.
    public:
        std::vector<Synapse> synapses;              // Synapses in the network

        /**
         * @brief Add a node to the network.
         * @param id Unique identifier for the node.
         * @param type Node type: "input", "hidden", or "output".
         * @param value Initial value for the node.
         */
        void addN(const std::string& id, const std::string& type, float value);

        /**
         * @brief Add a synapse to the network.
         * @param src Source node ID.
         * @param dest Destination node ID.
         * @param weight Initial weight of the synapse.
         */
        void addS(const std::string& src, const std::string& dest, float weight);

        void fwd(); // Perform forward propagation through the network.
        void validate(); // Validate the network structure. Ensures nodes and synapses are valid and checks for cycles.

        /**
         * @brief Calculate the average weight of all synapses in the network.
         * @return The average synaptic weight.
         */
        float avgWeight() const;

        /**
         * @brief Add noise to synaptic weights to introduce stochastic variability.
         * @param noiseLevel Maximum amplitude of the noise.
         */
        void addWeightNoise(float noiseLevel);

        void print() const; // Print the structure of the network. Outputs all nodes and synapses to the console.
    };
} // namespace N3R

#endif // N3R_H

