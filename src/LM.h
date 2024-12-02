#ifndef LM_H
#define LM_H

#include <string>
#include <unordered_map>
#include <vector>

namespace LM {

void init(size_t dim);                     // Initialize embeddings and matrix
void bldMtx(const std::string& path, size_t win); // Build co-occurrence matrix
void trn(size_t epochs);                   // Train embeddings
void norm();                               // Normalize embeddings
std::vector<float> getVec(const std::string& word);   // Get embedding vector
void addWrd(const std::string& word);      // Add new word to vocabulary
void save(const std::string& path);        // Save embeddings
void load(const std::string& path);        // Load embeddings
void trnAdpt(const std::string& path, size_t epochs); // Train with adaptive params
void expDim(size_t newDim);                // Expand dimensions dynamically
void initPrm(const std::string& path);     // Init params dynamically

} // namespace LM

#endif // LM_H

