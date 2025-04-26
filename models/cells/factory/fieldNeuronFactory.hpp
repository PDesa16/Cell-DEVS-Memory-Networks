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
    std::vector<Eigen::VectorXd> storedPatternsSource;

public:
    FieldNeuronFactory(const std::vector<Eigen::VectorXd>& source)
        : storedPatternsSource(source) {}

    std::shared_ptr<GridCell<FieldState, double>> create(
        const coordinates& cellId,
        const std::shared_ptr<const GridCellConfig<FieldState, double>>& cellConfig
    ) const override {
        auto modifiableConfig = std::make_shared<GridCellConfig<FieldState, double>>(*cellConfig);

        FieldState state;
        int numPatterns = 1; 
        int width = modifiableConfig->scenario->shape[0];
        int height = modifiableConfig->scenario->shape[1];
        int flatIndex = cellId[0] * width + cellId[1];

        state.storedPatterns = Eigen::VectorXd(numPatterns);
        state.storedPatterns(0) = storedPatternsSource[0](flatIndex);

        // Randomize direction (x, y, z)
        state.state(0) = RandomNumberGeneratorDEVS::generateUniformDelay(0.0, 1.0);
        state.state(1) = RandomNumberGeneratorDEVS::generateUniformDelay(0.0, 1.0);
        state.state(2) = RandomNumberGeneratorDEVS::generateUniformDelay(0.0, 1.0);

        // Set magnitude
        state.state(3) = storedPatternsSource[0](flatIndex);

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
