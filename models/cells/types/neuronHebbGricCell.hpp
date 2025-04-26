#ifndef NEURON_HEBB_GRID_CELL_HPP
#define NEURON_HEBB_GRID_CELL_HPP

#include <cmath>
#include "neuronState.hpp"
#include "../../../types/imageStructures.hpp"
#include "../../../utils/generalUtils.hpp"
#include "../../../utils/stochastic/random.hpp"
#include "../neuronBaseGridCell.hpp"
#include "../states/HebbState.hpp"
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>

using namespace cadmium::celldevs;

class NeuronHebbGridCell : public NeuronBaseGridCell<HebbState> {
public:
    using NeuronBaseGridCell::NeuronBaseGridCell;

    bool shouldTrain(const HebbState& state) const override {
        return state.training;
    }

    void trainingNeuron(
        const std::unordered_map<std::vector<int>, NeighborData<HebbState, double>>& neighborhood,
        HebbState& state
    ) const override {
        // Hebbian learning rule: \Delta w_ij = eta * s_i * x_j
        for (const auto& [neighborId, neighborData] : neighborhood) {
            double s_i = state.activationStrength;
            double x_j = neighborData.state->activationStrength;
            state.weights[neighborId] += state.learningRate * s_i * x_j;
        }
    }

    void updateCellState(
        const std::unordered_map<std::vector<int>, NeighborData<HebbState, double>>& neighborhood,
        HebbState& state
    ) const override {
        // Hebbian update rule: s_i(t+1) = \sum_j w_ij * x_j(t)
        double newActivation = 0.0;
        for (const auto& [neighborId, neighborData] : neighborhood) {
            double x_j = neighborData.state->activationStrength;
            double w_ij = state.weights[neighborId];
            newActivation += w_ij * x_j;
        }
        state.activationStrength = newActivation;
    }

    double GetEnergy(
        const std::unordered_map<std::vector<int>, NeighborData<HebbState, double>>& neighborhood,
        const HebbState& state
    ) const override{
        // Energy function: E = - \sum_{i,j} w_ij * x_j * s_i
        double energy = 0.0;
        for (const auto& [neighborId, neighborData] : neighborhood) {
            double x_j = neighborData.state->activationStrength;
            double s_i = state.activationStrength;
            double w_ij = state.weights.at(neighborId);
            energy -= w_ij * x_j * s_i;
        }
        return energy;
    }

    double outputDelay(const HebbState& state) const override {
        return state.time; 
    }
};

#endif // NEURON_HOPFIELD_GRID_CELL_HPP
