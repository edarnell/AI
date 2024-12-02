#include <iostream>
#include <cmath>
#include "N3R.h"

namespace N3R {

NNet::NNet() {
    std::cout << "Initializing Neural Network...\n";
}

void NNet::addN(const std::string& n, double w) {
    if (ns.find(n) == ns.end()) {
        ns[n] = {n, w};
    }
}

void NNet::addS(const std::string& s, const std::string& t, double w, double d) {
    ss.push_back({s, t, w, d});
}

void NNet::fwd(double I, int n) {
    double dT = std::exp(-0.01 * n);
    for (auto& [id, node] : ns) {
        double adj = I * dT / (1.0 + node.w);
        node.w = std::clamp(node.w + adj, 0.0, 1.0);
    }
}

void NNet::chkCntr() const {
    for (const auto& [id, node] : ns) {
        if (node.w == 0.0 && node.w > 0.0) {
            std::cout << "Contradiction detected in node: " << id << std::endl;
        }
    }
}

void NNet::chkMx() const {
    for (const auto& syn : ss) {
        if (syn.w > 0.5 && syn.d < 0.1) {
            std::cout << "Mutual exclusivity issue: " << syn.s << " -> " << syn.t << std::endl;
        }
    }
}

void NNet::chkLowX(double thr) const {
    for (const auto& syn : ss) {
        if (syn.w < thr) {
            std::cout << "Low-confidence synapse: " << syn.s << " -> " << syn.t
                      << " (Weight: " << syn.w << ")" << std::endl;
        }
    }
}

void NNet::chkConf(double confThr) const {
    for (const auto& syn : ss) {
        double threshold = calcThr(syn, confThr);
        if (syn.w > threshold) {
            std::cout << "Overconfidence detected in synapse: " << syn.s << " -> " << syn.t
                      << " (Weight: " << syn.w << ", Threshold: " << threshold << ")" << std::endl;
        }
    }
}

void NNet::chkAnth(double weightThr, double distThr) const {
    for (const auto& syn : ss) {
        if (syn.w > weightThr && syn.d < distThr) {
            std::cout << "Potential anthropocentric bias in synapse: " << syn.s << " -> " << syn.t
                      << " (Weight: " << syn.w << ", Distance: " << syn.d << ")" << std::endl;
        }
    }
}

double NNet::calcThr(const S3R& syn, double confThr) const {
    double propagatedUncertainty = syn.w * (1.0 - syn.f);
    return 1.0 - propagatedUncertainty - confThr;
}

void NNet::validate(VM mode, double confThr, double lowXThr, double weightThr, double distThr) const {
    switch (mode) {
        case Node:
            // Node-level checks
            chkCntr();
            chkMx();
            break;

        case Synapse:
            // Synapse-level checks
            chkLowX(lowXThr);
            chkConf(confThr);
            chkAnth(weightThr, distThr);
            break;

        case Model:
            // Combined checks for overall model validation
            chkCntr();
            chkMx();
            chkLowX(lowXThr);
            chkConf(confThr);
            chkAnth(weightThr, distThr);
            break;

        default:
            std::cout << "Unknown validation mode!" << std::endl;
    }
}

void NNet::validateModel(double confThr, double lowXThr, double weightThr, double distThr) const {
    std::cout << "Validating entire model...\n";

    // Validate individual nodes
    for (const auto& [id, node] : ns) {
        validateNode(node);
    }

    // Validate individual synapses
    for (const auto& syn : ss) {
        validateSynapse(syn, confThr, lowXThr, weightThr, distThr);
    }

    // Global Check: Ensure all nodes are connected
    std::cout << "Checking network connectivity...\n";
    if (!isConnected()) {
        std::cout << "Warning: Network is not fully connected!" << std::endl;
    }

    // Global Check: Detect cyclic dependencies
    std::cout << "Checking for cyclic dependencies...\n";
    if (hasCycle()) {
        std::cout << "Warning: Cyclic dependencies detected!" << std::endl;
    }

    // Global Check: Aggregate bias
    double avgWeight = calculateAverageWeight();
    std::cout << "Average synapse weight: " << avgWeight << std::endl;

    std::cout << "Model validation complete.\n";
}

// Helper: Check network connectivity
bool NNet::isConnected() const {
    std::unordered_map<std::string, bool> visited;
    for (const auto& [id, node] : ns) {
        visited[id] = false;
    }

    if (visited.empty()) return true; // No nodes = trivially connected

    // Start DFS from the first node
    auto it = visited.begin();
    dfs(it->first, visited);

    // If any node is unvisited, the network is disconnected
    for (const auto& [id, wasVisited] : visited) {
        if (!wasVisited) return false;
    }

    return true;
}

void NNet::dfs(const std::string& id, std::unordered_map<std::string, bool>& visited) const {
    if (visited[id]) return; // Already visited
    visited[id] = true;

    // Explore neighbors
    for (const auto& syn : ss) {
        if (syn.s == id) dfs(syn.t, visited);
    }
}

// Helper: Detect cyclic dependencies
bool NNet::hasCycle() const {
    std::unordered_map<std::string, bool> visited;
    std::unordered_map<std::string, bool> stack;

    for (const auto& [id, node] : ns) {
        if (hasCycleUtil(id, visited, stack)) {
            return true;
        }
    }
    return false;
}

bool NNet::hasCycleUtil(const std::string& id, std::unordered_map<std::string, bool>& visited, std::unordered_map<std::string, bool>& stack) const {
    if (!visited[id]) {
        visited[id] = true;
        stack[id] = true;

        for (const auto& syn : ss) {
            if (syn.s == id) {
                if (!visited[syn.t] && hasCycleUtil(syn.t, visited, stack)) {
                    return true;
                }
                if (stack[syn.t]) return true;
            }
        }
    }

    stack[id] = false;
    return false;
}

// Helper: Calculate average weight
double NNet::calculateAverageWeight() const {
    double totalWeight = 0.0;
    for (const auto& syn : ss) {
        totalWeight += syn.w;
    }
    return ss.empty() ? 0.0 : totalWeight / ss.size();
}

} // namespace N3R
