#include "LM.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace LM {

// Storage
namespace {
std::unordered_map<std::string, std::unordered_map<std::string, float>> mtx; // Co-occurrence matrix
std::unordered_map<std::string, std::vector<float>> emb; // Word and context embeddings
size_t dim = 50;       // Embedding dimension
float lr = 0.01f;      // Learning rate
float decay = 0.001f;  // Decay rate

// Utilities
std::vector<std::string> tokenize(const std::string& text);
std::string toLower(const std::string& text);
std::string stripPunct(const std::string& text);
bool isContext(const std::string& token);
}

// Initialize
void init(size_t d) {
    dim = d;
    mtx.clear();
    emb.clear();
}

// Build co-occurrence matrix
void bldMtx(const std::string& path, size_t win) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Error: Failed to open file for co-occurrence matrix.");

    std::string line;
    while (std::getline(file, line)) {
        auto tokens = tokenize(stripPunct(toLower(line)));
        for (size_t i = 0; i < tokens.size(); ++i) {
            for (size_t j = std::max(0, static_cast<int>(i) - static_cast<int>(win));
                 j < std::min(tokens.size(), i + win + 1); ++j) {
                if (i != j) mtx[tokens[i]][tokens[j]] += 1.0f;
            }
        }
    }
}

// Add context embedding
void addContextEmbedding(const std::string& context) {
    if (!emb.count(context)) {
        emb[context].resize(dim);
        std::generate(emb[context].begin(), emb[context].end(),
                      []() { return static_cast<float>(rand()) / RAND_MAX; });
    }
}

// Pool embeddings dynamically for contexts
std::vector<float> getContextEmbedding(const std::vector<std::string>& contexts) {
    std::vector<float> pooled(dim, 0.0f);
    for (const auto& ctx : contexts) {
        if (emb.count(ctx)) {
            for (size_t i = 0; i < dim; ++i) {
                pooled[i] += emb[ctx][i];
            }
        }
    }
    for (float& val : pooled) val /= contexts.size(); // Average pooled embedding
    return pooled;
}

// Relationship-driven update with context awareness
void updWithContext(const std::vector<std::string>& contexts, const std::string& wrd, const std::string& ctx, float cnt) {
    auto contextEmbedding = getContextEmbedding(contexts);
    auto& wVec = emb[wrd];
    auto& cVec = emb[ctx];
    for (size_t i = 0; i < dim; ++i) {
        float delta = lr * (contextEmbedding[i] * wVec[i] * cVec[i] - decay);
        float noise = static_cast<float>(rand()) / RAND_MAX * decay;
        wVec[i] = std::clamp(wVec[i] + delta + noise, 0.0f, 1.0f);
        cVec[i] = std::clamp(cVec[i] + delta + noise, 0.0f, 1.0f);
    }
}

// Competitive learning with recovery
void competitiveUpdate() {
    for (auto& [wrd, vec] : emb) {
        float maxWeight = *std::max_element(vec.begin(), vec.end());
        for (float& v : vec) {
            v = (v == maxWeight) ? v + lr : v * (1.0f - decay) + decay * 0.01f; // Recovery term
        }
    }
}

// Train with context
void trnWithContext(const std::vector<std::string>& contexts, size_t epochs) {
    for (size_t e = 0; e < epochs; ++e) {
        for (const auto& [wrd, ctxs] : mtx) {
            for (const auto& [ctx, cnt] : ctxs) {
                if (emb.find(ctx) != emb.end()) {
                    updWithContext(contexts, wrd, ctx, log(1 + cnt));
                }
            }
        }
        competitiveUpdate(); // Apply competitive pruning after each epoch
    }
    norm(); // Normalize embeddings after training
}

// Normalize embeddings with drift
void norm() {
    for (auto& [wrd, vec] : emb) {
        float mag = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
        if (mag > 0) {
            for (float& v : vec) {
                v /= mag;
                v += static_cast<float>(rand()) / RAND_MAX * 0.01f; // Add minor drift
            }
        }
    }
}

// Add word
void addWrd(const std::string& wrd) {
    if (!emb.count(wrd)) {
        emb[wrd].resize(dim);
        std::generate(emb[wrd].begin(), emb[wrd].end(), []() { return static_cast<float>(rand()) / RAND_MAX; });
    }
}

// Save embeddings
void save(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Error: Failed to save embeddings.");
    for (const auto& [wrd, vec] : emb) {
        file << wrd;
        for (float v : vec) file << " " << v;
        file << "\n";
    }
    for (const auto& [ctx, vec] : emb) {
        if (isContext(ctx)) {
            file << "CTX " << ctx;
            for (float v : vec) file << " " << v;
            file << "\n";
        }
    }
}

// Load embeddings
void load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Error: Failed to load embeddings.");
    std::string line, token;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        stream >> token;
        if (token == "CTX") {
            std::string ctx;
            stream >> ctx;
            std::vector<float> vec(dim, 0.0f);
            for (float& v : vec) stream >> v;
            emb[ctx] = vec;
        } else {
            std::string wrd = token;
            std::vector<float> vec(dim, 0.0f);
            for (float& v : vec) stream >> v;
            emb[wrd] = vec;
        }
    }
}

// Utility functions
namespace {
std::vector<std::string> tokenize(const std::string& txt) {
    std::istringstream stream(txt);
    std::vector<std::string> tokens;
    std::string tok;
    while (stream >> tok) tokens.push_back(tok);
    return tokens;
}

std::string toLower(const std::string& txt) {
    std::string res = txt;
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

std::string stripPunct(const std::string& txt) {
    std::string res;
    std::remove_copy_if(txt.begin(), txt.end(), std::back_inserter(res), ::ispunct);
    return res;
}

bool isContext(const std::string& token) {
    return token.find("CTX") == 0;
}
}

} // namespace LM




