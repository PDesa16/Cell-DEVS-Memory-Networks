#ifndef NEURON_MODERN_HOPFIELD_GRID_CELL_HPP
#define NEURON_MODERN_HOPFIELD_GRID_CELL_HPP

#include <cmath>
#include "../../../types/imageStructures.hpp"
#include "../../../utils/generalUtils.hpp"
#include "../../../utils/stochastic/random.hpp"
#include "../neuronBaseGridCell.hpp"
#include "../states/modernHopfieldState.hpp"
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>

using namespace cadmium::celldevs;

class NeuronModernHopfieldGridCell : public NeuronBaseGridCell<ModernHopfieldState> {
public:
    using NeuronBaseGridCell::NeuronBaseGridCell;

    bool shouldTrain(const ModernHopfieldState& state) const override {
        return state.training;
    }

    void trainingNeuron(
        const std::unordered_map<std::vector<int>, NeighborData<ModernHopfieldState, double>>& neighborhood,
        ModernHopfieldState& state
    ) const override {
        // Training logic 
    }


    void updateCellState(
        const std::unordered_map<std::vector<int>, NeighborData<ModernHopfieldState, double>>& neighborhood,
        ModernHopfieldState& state
    ) const override {
        double beta = 1.0;
    
        if (state.globalContext) {
            // --- Global Update ---
            int featureDim = state.storedPatterns.cols();
            int numPatterns = state.storedPatterns.rows();
    
            Eigen::VectorXd scores = beta * (state.storedPatterns.transpose() * state.activationStrength); // (features x 1)
    
            double max_score = scores.maxCoeff();
            Eigen::VectorXd exp_scores = (scores.array() - max_score).exp();
            double sum_exp = exp_scores.sum();
            if (sum_exp == 0) sum_exp = 1e-8; // Prevent divide-by-zero
            Eigen::VectorXd softmax_weights = exp_scores / sum_exp;
    
            Eigen::VectorXd updated_xi = softmax_weights;
    
            state.activationStrength = updated_xi.norm();
        } else {
            // --- Local Update ---
            int featureDim = state.storedPatterns.cols();
            int numPatterns = state.storedPatterns.rows();
            int neighborCount = static_cast<int>(neighborhood.size());
            int totalCount = neighborCount + 1;
    
            if (totalCount == 0) return;
    
            Eigen::MatrixXd subsetPatterns(totalCount, featureDim);
            Eigen::VectorXd activationsNeighborhood(totalCount);
    
            int idx = 0;
            for (const auto& [pos, neighbor] : neighborhood) {
                activationsNeighborhood(idx) = neighbor.state->activationStrength;
                int flatten_index = neighbor.state->coords[1] * state.imageWidth + neighbor.state->coords[0];
    
                if (flatten_index >= 0 && flatten_index < numPatterns) {
                    subsetPatterns.row(idx) = state.storedPatterns.row(flatten_index);
                } else {
                    subsetPatterns.row(idx).setZero();
                }
    
                idx++;
            }
    
            activationsNeighborhood(idx) = state.activationStrength;
            int self_flatten_index = state.coords[1] * state.imageWidth + state.coords[0];
    
            if (self_flatten_index >= 0 && self_flatten_index < numPatterns) {
                subsetPatterns.row(idx) = state.storedPatterns.row(self_flatten_index);
            } else {
                subsetPatterns.row(idx).setZero();
            }
    
            Eigen::VectorXd scores = beta * (subsetPatterns.transpose() * activationsNeighborhood); // (features x 1)
    
            double max_score = scores.maxCoeff();
            Eigen::VectorXd exp_scores = (scores.array() - max_score).exp();
            double sum_exp = exp_scores.sum();
            if (sum_exp == 0) sum_exp = 1e-8;
            Eigen::VectorXd softmax_weights = exp_scores / sum_exp;
    
            Eigen::VectorXd updated_xi = softmax_weights;
    
            state.activationStrength = updated_xi.norm();
        }
    }


    double GetEnergy(
        const std::unordered_map<std::vector<int>, NeighborData<ModernHopfieldState, double>>& neighborhood,
        const ModernHopfieldState& state
    ) const override {
        int neighborhoodSize = neighborhood.size() + 1; 
        Eigen::VectorXd xi(neighborhoodSize);

        int selfIndex = state.coords[0] * state.imageWidth + state.coords[1];
        xi(0) = state.activationStrength;

        int idx = 1;
        for (const auto& [_, neighborData] : neighborhood) {
            xi(idx) = neighborData.state->activationStrength;
            ++idx;
        }

        double beta = 1.0;
        double norm_sq = xi.squaredNorm();

        double partition = 0.0;
        for (int i = 0; i < xi.size(); ++i) {
            partition += std::exp(beta * xi.dot(xi)); 
        }

        double energy = -(1.0 / beta) * std::log(partition) + 0.5 * norm_sq;
        return energy;
    }

    double outputDelay(const ModernHopfieldState& state) const override {
        return state.time; 
    }
};

#endif // NEURON_MODERN_HOPFIELD_GRID_CELL_HPP
