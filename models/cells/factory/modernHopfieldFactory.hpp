#ifndef MODERN_HOPFIELD_NEURON_FACTORY_HPP
#define MODERN_HOPFIELD_NEURON_FACTORY_HPP

#include <memory>
#include <Eigen/Dense>
#include "../../../utils/stochastic/random.hpp"
#include "../states/modernHopfieldState.hpp"
#include "../types/neuronModernHopfieldGridCell.hpp"
#include "gridCellFactoryBase.hpp"

using namespace cadmium::celldevs;

class ModernHopfieldNeuronFactory : public GridCellFactoryBase<ModernHopfieldState> {
private:
    Eigen::MatrixXd storedPatternsSource;  // (numPatterns x imageSize)
    Eigen::MatrixXd queryInput;             // (optional, blurred query image)
    bool useQueryInput;

public:
    ModernHopfieldNeuronFactory(const Eigen::MatrixXd& source,
                                 const Eigen::MatrixXd& query = Eigen::MatrixXd(),
                                 bool useQuery = false)
        : storedPatternsSource(source), queryInput(query), useQueryInput(useQuery) {}

        std::shared_ptr<GridCell<ModernHopfieldState, double>> create(
            const coordinates& cellId,
            const std::shared_ptr<const GridCellConfig<ModernHopfieldState, double>>& cellConfig
        ) const override {
            auto modifiableConfig = std::make_shared<GridCellConfig<ModernHopfieldState, double>>(*cellConfig);
        
            ModernHopfieldState state;
            int width = modifiableConfig->scenario->shape[0];
            int height = modifiableConfig->scenario->shape[1];
            int flatIndex = cellId[0] * width + cellId[1]; // (row * width + col)
        
            int numPatterns = storedPatternsSource.cols(); // **columns = patterns**
        
            // Set activation strength (query input or memory)
            if (useQueryInput && queryInput.size() > 0) {
                state.activationStrength = queryInput(flatIndex, 0); // (row, col)
            } else {
                state.activationStrength = storedPatternsSource(flatIndex, 0); // (row = pixel index, col = pattern 0)
            }
        
            // Store all pattern values for this pixel
            state.storedPatterns = storedPatternsSource.row(flatIndex).transpose(); 
            // row(flatIndex) = all patterns for this pixel
        
            state.coords = {cellId[0], cellId[1]};
            state.imageWidth = width;
            state.imageHeight = height;
        
            modifiableConfig->state = state;
        
            return std::make_shared<NeuronModernHopfieldGridCell>(
                cellId,
                modifiableConfig,
                std::make_shared<ModernHopfieldState>(state)
            );
        }        
};

#endif // MODERN_HOPFIELD_NEURON_FACTORY_HPP
