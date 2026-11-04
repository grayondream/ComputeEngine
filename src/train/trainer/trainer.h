#pragma once

#include <string>
#include <unordered_map>

#include "core/base/error.h"
#include "core/ir/graph.h"
#include "train/data/dataloader_plugin.h"

namespace ie {

// Evaluation metrics produced by Trainer::evaluate. The concrete metric set is
// deliberately open; loss/accuracy cover the common supervised case.
struct Metrics {
    double loss{0.0};
    double accuracy{0.0};
    std::unordered_map<std::string, double> values;
};

// Training loop contract (docs/readme.md §9.3). P0.5 freezes the surface only;
// the loop itself arrives with the training milestone.
class Trainer {
public:
    struct Config {
        int epochs{10};
        int batch_size{32};
        float lr{1e-4};
        std::string optimizer{"adamw"};
        std::string scheduler{"cosine"};
        std::string loss{"cross_entropy"};
        bool use_amp{false};
        int grad_accum_steps{1};
        int checkpoint_interval{1000};
        std::string checkpoint_dir;
    };

    Status train(Graph& model, DataloaderPlugin& dataloader, const Config& config) {
        (void)model;
        (void)dataloader;
        (void)config;
        return Error::unsupported("training not implemented in P0.5");
    }

    Result<Metrics> evaluate(Graph& model, DataloaderPlugin& dataloader) {
        (void)model;
        (void)dataloader;
        return Error::unsupported("evaluation not implemented in P0.5");
    }

    Status resume(const std::string& checkpoint_path) {
        (void)checkpoint_path;
        return Error::unsupported("training resume not implemented in P0.5");
    }
};

}  // namespace ie
