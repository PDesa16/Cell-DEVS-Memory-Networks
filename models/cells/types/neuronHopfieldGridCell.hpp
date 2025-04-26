#ifndef NEURON_HOPFIELD_GRID_CELL_HPP
#define NEURON_HOPFIELD_GRID_CELL_HPP

#include <cmath>
#include "../../../types/imageStructures.hpp"
#include "../../../utils/generalUtils.hpp"
#include "../../../utils/stochastic/random.hpp"
#include "../neuronBaseGridCell.hpp"
#include "../states/hopfieldState.hpp"
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>

using namespace cadmium::celldevs;

class NeuronHopfieldGridCell : public NeuronBaseGridCell<HopfieldState> {
public:
    using NeuronBaseGridCell::NeuronBaseGridCell;

    bool shouldTrain(const HopfieldState& state) const override {
        return false;
    }

    void trainingNeuron(
        const std::unordered_map<std::vector<int>, NeighborData<HopfieldState, double>>& neighborhood,
        HopfieldState& state
    ) const override {
        // No Training Logic - Using globalWeights directly
    }

    void updateCellState(
        const std::unordered_map<std::vector<int>, NeighborData<HopfieldState, double>>& neighborhood,
        HopfieldState& state
    ) const override {
        auto [x, y] = GeneralUtils::stringToIndices(this->getId());
        double sum = 0.0;
        int selfFlatIndex = x * state.imageWidth + y;

        for (const auto& [neighborId, neighborData] : neighborhood) {
            int neighborState = neighborData.state->activationStatus;
            int neighborFlatIndex = neighborId[0] * state.imageWidth + neighborId[1];

            if (neighborId != std::vector<int>{x, y}) {
                double weight = state.globalWeights(selfFlatIndex, neighborFlatIndex);
                sum += weight * static_cast<double>(neighborState);
            }
        }

        state.activationStatus = GeneralUtils::signum(sum);
        state.neighboringStates->stateMatrix(x, y) = state.activationStatus;
    }

    double GetEnergy(
        const std::unordered_map<std::vector<int>, NeighborData<HopfieldState, double>>& neighborhood,
        const HopfieldState& state
    ) const override {
        double energy = 0.0;
        auto [x, y] = GeneralUtils::stringToIndices(this->getId());
        int selfFlatIndex = x * state.imageWidth + y;
        int s_i = state.activationStatus;

        for (const auto& [neighborId, neighborData] : neighborhood) {
            int neighborFlatIndex = neighborId[0] * state.imageWidth + neighborId[1];
            int s_j = neighborData.state->activationStatus;

            double w_ij = state.globalWeights(selfFlatIndex, neighborFlatIndex);
            energy -= 0.5 * w_ij * s_j * s_i;
        }

        return energy;
    }

    double outputDelay(const HopfieldState& state) const override {
        return state.time;
    }
};

#endif // NEURON_HOPFIELD_GRID_CELL_HPP
