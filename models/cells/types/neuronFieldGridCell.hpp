#ifndef NEURON_FIELD_GRID_CELL_HPP
#define NEURON_FIELD_GRID_CELL_HPP

#include <cmath>
#include "../../../types/imageStructures.hpp"
#include "../../../utils/generalUtils.hpp"
#include "../../../utils/stochastic/random.hpp"
#include "../neuronBaseGridCell.hpp"
#include "../states/FieldState.hpp"
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>

using namespace cadmium::celldevs;

class NeuronFieldGridCell : public NeuronBaseGridCell<FieldState> {
public:
    using NeuronBaseGridCell::NeuronBaseGridCell;

    bool shouldTrain(const FieldState& state) const override {
        return state.training;
    }

    void trainingNeuron(
        const std::unordered_map<std::vector<int>, NeighborData<FieldState, double>>& neighborhood,
        FieldState& state
    ) const override {
        Eigen::Vector3d currentPos = state.state.head<3>();
        double currentMag = state.state(3);
        Eigen::Vector3d netForce = Eigen::Vector3d::Zero();

        for (const auto& [coord, neighbor] : neighborhood) {
                Eigen::Vector3d neighborPos = neighbor.state->state.head<3>();
                double neighborMag = neighbor.state->state(3);
                Eigen::Vector3d delta = neighborPos - currentPos;
                double distance = delta.norm();

                if (distance > 1e-6) {
                    double invDistSq = 1.0 / (distance * distance + 1e-6);
                    Eigen::Vector3d unitDelta = delta.normalized();
                    double alignment = currentPos.normalized().dot(neighborPos.normalized());

                    Eigen::Vector3d force = -currentMag * neighborMag * invDistSq * unitDelta;
                    netForce += alignment * force;
                }
        }
    
        netForce /= static_cast<double>(state.storedPatterns.size());

        Eigen::Vector3d updatedPos = currentPos + 0.1 * netForce;
        if (updatedPos.norm() > 1e-6) {
            updatedPos.normalize();
        }

        double movement = (updatedPos - currentPos).norm();
        state.state.head<3>() = updatedPos;

        const double epsilon = 1e-4;
        if (movement < epsilon) {
            state.time = std::numeric_limits<double>::infinity();
        }
    }

    void updateCellState(
        const std::unordered_map<std::vector<int>, NeighborData<FieldState, double>>& neighborhood,
        FieldState& state
    ) const override {
        Eigen::Vector3d currentPos = state.state.head<3>();
        double currentMag = state.state(3);
        Eigen::Vector3d vi = currentPos.normalized();
        double weightedSum = 0.0;

        for (const auto& [coord, neighbor] : neighborhood) {
            Eigen::Vector3d delta = neighbor.state->state.head<3>() - currentPos;
            double distance = delta.norm();
            if (distance > 1e-6) {
                Eigen::Vector3d vj = neighbor.state->state.head<3>().normalized();
                double alignment = -1 * vi.dot(vj);
                double mag_j = neighbor.state->state(3);
                double magDiff = currentMag - mag_j;
                double influence = std::tanh(2.0 * magDiff * alignment);
                weightedSum += influence;
            }
        }

        double beta = 0.1;
        state.state(3) = std::clamp(state.state(3) + beta * weightedSum , 0.0, 1.0);
    }

    double GetEnergy(
        const std::unordered_map<std::vector<int>, NeighborData<FieldState, double>>& neighborhood,
        const FieldState& state
    ) const override {
        if (state.training) {
            // Misalignment-based energy during training
            Eigen::Vector3d vi = state.state.head<3>().normalized();
            double energy = 0.0;
            for (const auto& [coord, neighbor] : neighborhood) {
                Eigen::Vector3d vj = neighbor.state->state.head<3>().normalized();
                double mag_j = neighbor.state->state(3);
                double alignment = std::abs(vi.dot(vj));
                energy += (1.0 - alignment) * mag_j;
            }
            return energy;
        } else {
            // Magnitude-change energy during recall
            double change = std::abs(state.state(3) - state.previousActivation);
            return change;
        }
    }

    double outputDelay(const FieldState& state) const override {
        return state.time; 
    }

    FieldState& getState() {
        return state;
    }
    
};

#endif // NEURON_HOPFIELD_GRID_CELL_HPP
