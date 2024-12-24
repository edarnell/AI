#include "N3R.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <unordered_set>
#include "utils.h"

namespace N3R {
    // Add a node to the network
    void NNet::addN(const std::string& id, const std::string& type, float value) {
        if (nodes.count(id)) throw std::runtime_error("Error: Duplicate node ID.");
        nodes[id] = Node{id, type, value};
    }

    // Add a synapse to the network
    void NNet::addS(const std::string& src, const std::string& dest, float weight) {
        if (!nodes.count(src) || !nodes.count(dest))
            throw std::runtime_error("Error: Undefined source or destination node.");
        synapses.push_back(Synapse{src, dest, weight + randomFloat()}); // Add variability to weight
    }

    // Forward propagate through the network
    void NNet::fwd() {
        // Reset all node values before propagation
        for (auto& [id, node] : nodes) {
            node.value = (node.type == "input") ? node.value : 0.0f;
        }

        // Perform propagation
        for (const auto& syn : synapses) {
            auto& srcNode = nodes[syn.src];
            auto& destNode = nodes[syn.dest];
            float delta = srcNode.value * syn.weight;
            destNode.value += delta + randomFloat(); // Add stochastic variability
        }

        // Normalize node values
        for (auto& [id, node] : nodes) {
            node.value = std::tanh(node.value); // Ensure values stay bounded
        }
    }

    // Validate the network
    void NNet::validate() {
        validateNodes();
        validateSynapses();
        checkCycles();
    }

    void NNet::validateNodes() const {
        for (const auto& [id, node] : nodes) {
            if (node.type != "input" && node.type != "hidden" && node.type != "output")
                throw std::runtime_error("Error: Invalid node type.");
        }
    }

    void NNet::validateSynapses() const {
        for (const auto& syn : synapses) {
            if (!nodes.count(syn.src) || !nodes.count(syn.dest))
                throw std::runtime_error("Error: Undefined nodes in synapse.");
        }
    }

    void NNet::checkCycles() const {
        // Detect cycles using DFS
        std::unordered_map<std::string, bool> visited;
        std::unordered_map<std::string, bool> stack;

        for (const auto& [id, node] : nodes) {
            if (!visited[id] && dfsCycleCheck(id, visited, stack)) {
                throw std::runtime_error("Error: Cycle detected in the network.");
            }
        }
    }

    bool NNet::dfsCycleCheck(const std::string& nodeId,
                             std::unordered_map<std::string, bool>& visited,
                             std::unordered_map<std::string, bool>& stack) const {
        visited[nodeId] = true;
        stack[nodeId] = true;

        for (const auto& syn : synapses) {
            if (syn.src == nodeId) {
                if (!visited[syn.dest] && dfsCycleCheck(syn.dest, visited, stack))
                    return true;
                else if (stack[syn.dest])
                    return true;
            }
        }
        stack[nodeId] = false;
        return false;
    }

    // Calculate the average weight of all synapses
    float NNet::avgWeight() const {
        if (synapses.empty()) return 0.0f;
        float totalWeight = 0.0f;
        for (const auto& syn : synapses) {
            totalWeight += syn.weight;
        }
        return totalWeight / synapses.size();
    }

    // Introduce noise into synapse weights
    void NNet::addWeightNoise(float noiseLevel) {
        for (auto& syn : synapses) {
            syn.weight += randomFloat(-noiseLevel, noiseLevel);
        }
    }

    // Print network structure
    void NNet::print() const {
        for (const auto& [id, node] : nodes) {
            std::cout << "Node: " << id << ", Type: " << node.type
                      << ", Value: " << node.value << std::endl;
        }

        for (const auto& syn : synapses) {
            std::cout << "Synapse: " << syn.src << " -> " << syn.dest
                      << ", Weight: " << syn.weight << std::endl;
        }
    }

} // namespace N3R
