#ifndef LM_H
#define LM_H

#include <string>
#include <vector>
#include <unordered_map>

namespace LM {

/**
 * @brief Initialize the embedding system with a given dimension.
 * @param d The dimensionality of the embedding vectors.
 */
void init(size_t d);

/**
 * @brief Build the co-occurrence matrix from a text file.
 * @param path The path to the input file.
 * @param win The context window size for co-occurrence.
 */
void bldMtx(const std::string& path, size_t win);

/**
 * @brief Add a new context-specific embedding.
 * @param context The unique context identifier.
 */
void addContextEmbedding(const std::string& context);

/**
 * @brief Retrieve the pooled embedding vector for a set of contexts.
 * @param contexts The list of context identifiers.
 * @return A pooled embedding vector representing the contexts.
 */
std::vector<float> getContextEmbedding(const std::vector<std::string>& contexts);

/**
 * @brief Update word embeddings using a context-aware mechanism.
 * @param contexts The list of context identifiers.
 * @param wrd The target word.
 * @param ctx The co-occurring word.
 * @param cnt The co-occurrence count.
 */
void updWithContext(const std::vector<std::string>& contexts, const std::string& wrd, const std::string& ctx, float cnt);

/**
 * @brief Perform competitive updates to refine embeddings.
 */
void competitiveUpdate();

/**
 * @brief Train embeddings with context-awareness over specified epochs.
 * @param contexts The list of context identifiers.
 * @param epochs The number of training epochs.
 */
void trnWithContext(const std::vector<std::string>& contexts, size_t epochs);

/**
 * @brief Normalize all embeddings to maintain stability.
 */
void norm();

/**
 * @brief Add a new word to the embedding space with randomized initialization.
 * @param wrd The word to add.
 */
void addWrd(const std::string& wrd);

/**
 * @brief Save all embeddings (words and contexts) to a file.
 * @param path The path to the output file.
 */
void save(const std::string& path);

/**
 * @brief Load embeddings (words and contexts) from a file.
 * @param path The path to the input file.
 */
void load(const std::string& path);

} // namespace LM

#endif // LM_H


