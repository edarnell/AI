#include "LM.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>

namespace LM {

// Storage
namespace {
std::unordered_map<std::string, std::unordered_map<std::string, float>> mtx; // Co-occurrence matrix
std::unordered_map<std::string, std::vector<float>> emb; // Word vectors
size_t dim = 50;       // Embedding dimension
float lr = 0.01f;      // Learning rate
float decay = 0.001f;  // Decay rate

// Utilities
std::vector<std::string> tokenize(const std::string& text);
std::string toLower(const std::string& text);
std::string stripPunct(const std::string& text);
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
    if (!file.is_open()) throw std::runtime_error("Failed to open file.");

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

// Hebbian update
void upd(const std::string& wrd, const std::string& ctx, float cnt) {
    auto& wVec = emb[wrd];
    auto& cVec = emb[ctx];
    for (size_t i = 0; i < dim; ++i) {
        wVec[i] += lr * cnt * cVec[i];
        cVec[i] += lr * cnt * wVec[i];
        wVec[i] *= (1.0f - decay);
        cVec[i] *= (1.0f - decay);
    }
}

// Train
void trn(size_t epochs) {
    for (size_t e = 0; e < epochs; ++e) {
        for (const auto& [wrd, ctxs] : mtx) {
            for (const auto& [ctx, cnt] : ctxs) {
                if (emb.find(ctx) != emb.end()) {
                    upd(wrd, ctx, log(1 + cnt));
                }
            }
        }
    }
}

// Normalize
void norm() {
    for (auto& [wrd, vec] : emb) {
        float mag = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
        if (mag > 0) for (float& v : vec) v /= mag;
    }
}

// Get vector
std::vector<float> getVec(const std::string& wrd) {
    return emb.count(wrd) ? emb[wrd] : std::vector<float>(dim, 0.0f);
}

// Add word
void addWrd(const std::string& wrd) {
    if (!emb.count(wrd)) emb[wrd] = std::vector<float>(dim, 0.1f);
}

// Save
void save(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Failed to save embeddings.");
    for (const auto& [wrd, vec] : emb) {
        file << wrd;
        for (float v : vec) file << " " << v;
        file << "\n";
    }
}

// Load
void load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Failed to load embeddings.");
    std::string line, wrd;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        stream >> wrd;
        std::vector<float> vec(dim, 0.0f);
        for (float& v : vec) stream >> v;
        emb[wrd] = vec;
    }
}

// Adaptive training
void trnAdpt(const std::string& path, size_t epochs) {
    initPrm(path);
    trn(epochs);
}

// Expand dimensions
void expDim(size_t newDim) {
    if (newDim > dim) {
        for (auto& [wrd, vec] : emb) vec.resize(newDim, 0.1f);
        dim = newDim;
    }
}

// Initialize parameters
void initPrm(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Failed to open file for params.");

    std::unordered_set<std::string> vocab;
    size_t totalTokens = 0, totalLines = 0;
    std::string line;

    while (std::getline(file, line)) {
        auto tokens = tokenize(stripPunct(toLower(line)));
        totalTokens += tokens.size();
        totalLines++;
        vocab.insert(tokens.begin(), tokens.end());
    }

    size_t vocabSize = vocab.size();
    float avgLength = static_cast<float>(totalTokens) / totalLines;

    dim = std::min(300, static_cast<int>(std::sqrt(vocabSize * avgLength)));
    lr = 1.0f / std::sqrt(avgLength);
    decay = 0.01f / dim;
}

// Utilities
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
}

} // namespace LM


