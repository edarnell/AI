#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <bzlib.h>
#include "utils.h"  // Utility functions
#include "LM.h"     // Model 
#include "N3R.h"    // Neural Network logic
#include "Xi.h" 
#include "zip.h"


namespace Xi {

    // Global state for training and model management
    LM model(50, 0.01, 0.001, 0.01);
    const std::string fM = "data/model.bz2";
    const std::string fGPT = "data/conversations.json";

    N3R::NNet nnet;
    std::string sha; // Track last trained state
    
void Xi::trn3R(const std::vector<TrnData>& data, float l, float t, float alpha, float beta, int maxE) {
        if (data.empty()) {
            log(Log::ERROR, "Training data is empty. Cannot train.");
            throw std::runtime_error("Training data is empty. Cannot train.");
        }
        if (maxE <= 0) {
            log(Log::ERROR, "Maximum number of epochs must be greater than zero.");
            throw std::runtime_error("Maximum number of epochs must be greater than zero.");
        }

        float prevL = std::numeric_limits<float>::max(); // Previous loss for convergence check
        float loss = 0.0f; // Current loss

        for (int en = 0; en < maxE; ++en) {
            loss = 0.0f; // Reset loss for each epoch

            for (const auto& r : data) {
                auto tgtE = model.getE(r.tgt);
                auto ctxE = model.getE(r.ctx);

                // Update embeddings using plasticity
                model.updE(tgtE, ctxE, l, alpha);

                // Calculate loss
                float d = std::inner_product(tgtE.begin(), tgtE.end(), ctxE.begin(), 0.0f);
                loss += model.calcL(d, r.lbl);
            }

            // Apply forgetting (decay unused connections) every 5 epochs
            if (en % 5 == 0) model.dcyW(beta);

            // Log progress based on log level
            log(Log::DEBUG, "Epoch " + std::to_string(en + 1) + ": Loss = " + std::to_string(loss / data.size()));

            // Convergence check
            if (std::abs(prevL - loss) < t) {
                log(Log::INFO, "Convergence after " + std::to_string(en + 1) + " epochs.");
                break;
            }

            prevL = loss; // Update previous loss
        }
    }

    void loadModel(const std::string& f) {
        std::cout << "Loading model: " << f << std::endl;

        try {
            // Try to load the existing model
            load(f);
            std::cout << "Model loaded successfully." << std::endl;

            if (sha256(fGPT) != sha) {
                std::cout << "New data detected. Incremental training starting...\n";
                ldzJSON("data/gpt.zip", "conversations.json");
                train(10);
                sha = sha256(fGPT);
                save(f);
                std::cout << "Training complete. Model updated." << std::endl;
            } else {
                std::cout << "No data updates. Model is up to date.\n";
            }
        } catch (const std::runtime_error& e) {
            // If loading fails, train from gpt.zip and save the model
            std::cerr << "Error loading model: " << e.what() << "\nTraining new model from gpt.zip..." << std::endl;

            ldzJSON("data/gpt.zip", "conversations.json");
            train(10);
            save(f);
            std::cout << "New model trained and saved to " << f << std::endl;
        }
    }

    
    void load(const std::string& filePath) {
        BZFILE* file = BZ2_bzopen(filePath.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("Unable to open model file for reading: " + filePath);
        }

        constexpr int BUFFER_SIZE = 4096;
        char buffer[BUFFER_SIZE];
        std::string modelData;
        int bytesRead;

        while ((bytesRead = BZ2_bzread(file, buffer, BUFFER_SIZE)) > 0) {
            modelData.append(buffer, bytesRead);
        }

        BZ2_bzclose(file);

        // Deserialize model data (Assuming the model supports a deserialize method)
        model.deserialize(modelData);
        std::cout << "Model loaded successfully from " << filePath << std::endl;
    }
    
    void save(const std::string& filePath) {
        BZFILE* file = BZ2_bzopen(filePath.c_str(), "wb");
        if (!file) {
            throw std::runtime_error("Unable to open model file for writing: " + filePath);
        }

        // Serialize model data (Assuming the model supports a serialize method)
        std::string modelData = model.serialize();

        if (BZ2_bzwrite(file, modelData.data(), modelData.size()) < 0) {
            BZ2_bzclose(file);
            throw std::runtime_error("Error writing model data to file: " + filePath);
        }

        BZ2_bzclose(file);
        std::cout << "Model saved successfully to " << filePath << std::endl;
    }
    
    void ldzJSON(const std::string& zf, const std::string& fn, int ep, float lr, float t, float a, float b) {
        Zip z(zf);  // Open the zip file

        z.ext(fn, [&](const char* buf, size_t sz) {
            static std::string bufStr; // Accumulate data from chunks
            bufStr.append(buf, sz);

            try {
                std::unordered_map<std::string, std::string> nd; // Node data
                auto pcm = Utils::chat(bufStr, nd);  // Parent-child map

                std::vector<Xi::TrnData> td; // Training data
                for (const auto& [p, cList] : pcm) {
                    std::string pc = nd[p]; // Parent content
                    for (const auto& c : cList) {
                        std::string cc = nd[c]; // Child content
                        td.emplace_back(pc, cc, 1.0f);
                    }
                }

                // Train on this chunk
                if (!td.empty()) {
                    Xi::trn3R(td, lr, t, a, b, ep);
                    std::cout << "Trained on " << td.size() << " samples.\n";
                }

                bufStr.clear(); // Clear buffer after processing
            } catch (const std::exception&) {
                // JSON incomplete, continue accumulating chunks
            }
        });

        std::cout << "Training completed from " << fn << " in " << zf << ".\n";
    }

    std::string generateResponse(const std::string& userInput) {
        auto contextEmbedding = model.getContextEmbedding({userInput});
        nnet.addN(userInput, "input", 1.0f);

        std::string bestResponse;
        float maxWeight = -1.0f;

        for (const auto& syn : nnet.synapses) {
            if (syn.src == userInput && syn.weight > maxWeight) {
                maxWeight = syn.weight;
                bestResponse = syn.dest;
            }
        }

        return bestResponse.empty() ? "I don't know yet." : bestResponse;
    }

    void saveConversation(const std::string& topic, const std::vector<std::pair<int64_t, std::string>>& newMessages) {
        Utils::appendToBzip2(fGPT, topic, newMessages);
        std::cout << "Saved conversation to topic: " << topic << std::endl;
    }

}  // namespace Xi
