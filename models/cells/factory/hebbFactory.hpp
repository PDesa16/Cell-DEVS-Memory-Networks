#ifndef HEBB_NEURON_FACTORY_HPP
#define HEBB_NEURON_FACTORY_HPP

#include <memory>
#include <Eigen/Dense>
#include "../../../utils/stochastic/random.hpp"
#include "../states/HebbState.hpp"
#include "../types/neuronHebbGridCell.hpp"
#include "gridCellFactoryBase.hpp"

using namespace cadmium::celldevs;

class HebbNeuronFactory : public GridCellFactoryBase<HebbState> {
private:
    Eigen::MatrixXd storedPatternsSource;  
    Eigen::MatrixXd queryInput;             
    Eigen::MatrixXd trainingWeights;        
    mutable std::shared_ptr<std::unordered_map<std::vector<int>, double>> sharedWeights;
    bool trainingMode = true;               

public:
    // Constructor for training
    HebbNeuronFactory(const Eigen::MatrixXd& source,
                      const Eigen::MatrixXd& query = Eigen::MatrixXd())
        : storedPatternsSource(source), queryInput(query), trainingMode(true) {}

    // Constructor for recall
    HebbNeuronFactory(const Eigen::MatrixXd& source,
                      const Eigen::MatrixXd& query,
                      const Eigen::MatrixXd& trainingWeights_)
        : storedPatternsSource(source), queryInput(query), trainingWeights(trainingWeights_), trainingMode(false) {}

    std::shared_ptr<GridCell<HebbState, double>> create(
        const coordinates& cellId,
        const std::shared_ptr<const GridCellConfig<HebbState, double>>& cellConfig
    ) const override {
        auto modifiableConfig = std::make_shared<GridCellConfig<HebbState, double>>(*cellConfig);

        HebbState state;
        int width = modifiableConfig->scenario->shape[0];
        int height = modifiableConfig->scenario->shape[1];
        int flatIndex = cellId[0] * width + cellId[1]; 

        // Initialize shared weights if needed
        if (!sharedWeights) {
            sharedWeights = std::make_shared<std::unordered_map<std::vector<int>, double>>();

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int flatIdx = y * width + x;
                    if (trainingWeights.size() > 0) {
                        (*sharedWeights)[{x, y}] = trainingWeights(flatIdx, 0);
                    } else {
                        (*sharedWeights)[{x, y}] = 0.0;
                    }
                }
            }
        }

        // Set activationStrength depending on mode
        if (trainingMode) {
            state.activationStrength = storedPatternsSource.size() > 0 ? storedPatternsSource(flatIndex, 0) : 0.0;
        } else {
            state.training = false;
            state.activationStrength = queryInput.size() > 0 ? queryInput(flatIndex, 0) : 0.0;
        }

        state.coords = cellId;
        state.imageWidth = width;
        state.imageHeight = height;
        state.training = trainingMode;
        state.weights = sharedWeights;

        modifiableConfig->state = state;

        return std::make_shared<NeuronHebbGridCell>(
            cellId,
            modifiableConfig,
            std::make_shared<HebbState>(state)
        );
    }
};

#endif // HEBB_NEURON_FACTORY_HPP
