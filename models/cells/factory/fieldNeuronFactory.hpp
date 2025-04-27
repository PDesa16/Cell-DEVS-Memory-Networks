#ifndef FIELD_NEURON_FACTORY_HPP
#define FIELD_NEURON_FACTORY_HPP

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "../../../utils/stochastic/random.hpp"
#include "../states/FieldState.hpp"
#include "../types/neuronFieldGridCell.hpp"
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>
#include "gridCellFactoryBase.hpp"

using namespace cadmium::celldevs;

class FieldNeuronFactory : public GridCellFactoryBase<FieldState> {
private:
    Eigen::MatrixXd storedPatternsSource;
    Eigen::MatrixXd queryInput;         
    Eigen::MatrixXd trainingParameters; 
    bool useTrainingParameters = false;

public:
    // Regular constructor
    FieldNeuronFactory(
        const Eigen::MatrixXd& source,
        const Eigen::MatrixXd& query
    )
        : storedPatternsSource(source), queryInput(query) {}

    // Constructor with training parameters
    FieldNeuronFactory(
        const Eigen::MatrixXd& source,
        const Eigen::MatrixXd& query,
        const Eigen::MatrixXd& trainingParams
    )
        : storedPatternsSource(source),
          queryInput(query),
          trainingParameters(trainingParams),
          useTrainingParameters(true) {}

    std::shared_ptr<GridCell<FieldState, double>> create(
        const coordinates& cellId,
        const std::shared_ptr<const GridCellConfig<FieldState, double>>& cellConfig
    ) const override {
        auto modifiableConfig = std::make_shared<GridCellConfig<FieldState, double>>(*cellConfig);

        FieldState state;
        int width = modifiableConfig->scenario->shape[0];
        int height = modifiableConfig->scenario->shape[1];
        int flatIndex = cellId[0] * width + cellId[1]; 

        state.storedPatterns = storedPatternsSource;

        if (useTrainingParameters) {
            state.state = trainingParameters.row(flatIndex).transpose();
            state.state(3) = queryInput(flatIndex, 0);
            state.training = false;
        } else {
            // Randomize direction (x, y, z)
            state.state(0) = RandomNumberGeneratorDEVS::generateUniformDelay(0.0, 1.0);
            state.state(1) = RandomNumberGeneratorDEVS::generateUniformDelay(0.0, 1.0);
            state.state(2) = RandomNumberGeneratorDEVS::generateUniformDelay(0.0, 1.0);
            // Use actual image for training
            state.state(3) = storedPatternsSource(flatIndex,0);
        }

        state.coords = cellId;
        state.imageWidth = width;
        state.imageHeight = height;

        modifiableConfig->state = state;

        return std::make_shared<NeuronFieldGridCell>(
            cellId,
            modifiableConfig,
            std::make_shared<FieldState>(state)
        );
    }
};

#endif // FIELD_NEURON_FACTORY_HPP
