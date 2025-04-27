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
    }

    void updateCellState(
        const std::unordered_map<std::vector<int>, NeighborData<HopfieldState, double>>& neighborhood,
        HopfieldState& state
    ) const override {
        double sum = 0.0;
        double beta = 0.1;
        int selfFlatIndex = state.coords[0] * state.imageWidth + state.coords[1];

        for (const auto& [neighborId, neighborData] : neighborhood) {
            double neighborStrength = neighborData.state->activationStrength;
            int neighborFlatIndex = neighborData.state->coords[0] * state.imageWidth + neighborData.state->coords[1];
            double weight =  (*(state.weights))(selfFlatIndex, neighborFlatIndex);
            sum += weight * static_cast<double>(neighborStrength);
        }

        state.activationStrength += beta * GeneralUtils::sigmoid(sum);
    }

    double GetEnergy(
        const std::unordered_map<std::vector<int>, NeighborData<HopfieldState, double>>& neighborhood,
        const HopfieldState& state
    ) const override {
        double energy = 0.0;
        int selfFlatIndex = state.coords[0] * state.imageWidth + state.coords[1];
        double s_i = state.activationStrength;

        for (const auto& [neighborId, neighborData] : neighborhood) {
            int neighborFlatIndex = neighborData.state->coords[0] * state.imageWidth + neighborData.state->coords[1];
            double s_j = neighborData.state->activationStrength;

            double w_ij = (*(state.weights))(selfFlatIndex, neighborFlatIndex);
            energy -= 0.5 * w_ij * s_j * s_i;
        }

        return energy;
    }

    double outputDelay(const HopfieldState& state) const override {
        return state.time;
    }

    HopfieldState& getState() {
        return state;
    }
};

#endif // NEURON_HOPFIELD_GRID_CELL_HPP
