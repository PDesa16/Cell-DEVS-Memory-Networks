#ifndef HOPFIELD_NEURON_FACTORY_HPP
#define HOPFIELD_NEURON_FACTORY_HPP

#include <memory>
#include <Eigen/Dense>
#include "../../../utils/stochastic/random.hpp"
#include "../states/modernHopfieldState.hpp"
#include "../types/neuronHopfieldGridCell.hpp"
#include "gridCellFactoryBase.hpp"

using namespace cadmium::celldevs;

class HopfieldNeuronFactory : public GridCellFactoryBase<HopfieldState> {
private:
    Eigen::MatrixXd storedPatternsSource; 
    Eigen::MatrixXd globalWeights;            
    Eigen::MatrixXd queryInput;            

public:
    HopfieldNeuronFactory(const Eigen::MatrixXd& source, \
        const Eigen::MatrixXd& query = Eigen::MatrixXd())
        : storedPatternsSource(source), queryInput(query) {
                globalWeights = GeneralUtils::calculateWeightMatrix(source);
        }

        std::shared_ptr<GridCell<HopfieldState, double>> create(
            const coordinates& cellId,
            const std::shared_ptr<const GridCellConfig<HopfieldState, double>>& cellConfig
        ) const override {
            auto modifiableConfig = std::make_shared<GridCellConfig<HopfieldState, double>>(*cellConfig);
        
            HopfieldState state;
            int width = modifiableConfig->scenario->shape[0];
            int height = modifiableConfig->scenario->shape[1];
            int flatIndex = cellId[0] * width + cellId[1]; 
        
            int numPatterns = storedPatternsSource.cols(); 
        
            state.activationStrength = queryInput(flatIndex, 0); 

            state.coords = cellId;
            state.imageWidth = width;
            state.imageHeight = height;
            state.weights = &globalWeights;
        
            modifiableConfig->state = state;
        
            return std::make_shared<NeuronHopfieldGridCell>(
                cellId,
                modifiableConfig,
                std::make_shared<HopfieldState>(state)
            );
        }        
};

#endif // HOPFIELD_NEURON_FACTORY_HPP
