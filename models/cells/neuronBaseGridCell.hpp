#ifndef NEURON_BASE_GRID_CELL_HPP
#define NEURON_BASE_GRID_CELL_HPP

#include <memory>
#include <unordered_map>
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>
#include "neuronStateInterface.hpp"

using namespace cadmium::celldevs;

// Templated GridCell base, state-agnostic
class NeuronBaseGridCellBase {
public:
    virtual std::shared_ptr<IGenericState> GetState() const = 0;
    virtual void SetState(std::shared_ptr<IGenericState> newState) = 0;
    virtual ~NeuronBaseGridCellBase() = default;
};

template<typename StateT>
class NeuronBaseGridCell : public GridCell<StateT, double>, public NeuronBaseGridCellBase {
public:
    using GridCellType = GridCell<StateT, double>;

    // Constructor
    NeuronBaseGridCell(
        const std::vector<int>& id,
        const std::shared_ptr<const GridCellConfig<StateT, double>>& config,
        const std::shared_ptr<StateT>& initialState)
        : GridCellType(id, config)
    {
        this->state = *initialState;
    }

    NeuronBaseGridCell() = default;

    // Return the state as a base class pointer for casting at a later time
    std::shared_ptr<IGenericState> GetState() const override {
        return std::make_shared<StateT>(this->state);
    }

    void SetState(std::shared_ptr<IGenericState> newState) override {
        this->state = *std::static_pointer_cast<StateT>(newState);
    }

    // Main driver (non-overridable)
    [[nodiscard]]
    StateT localComputation(
        StateT state,
        const std::unordered_map<std::vector<int>, NeighborData<StateT, double>>& neighborhood
    ) const final {
        if (state.time !=0) {

            if (shouldTrain(state)) {
                trainingNeuron(neighborhood, state);
            }

            updateCellState(neighborhood, state);

            double energy = GetEnergy(neighborhood, state);
            if (std::abs(energy) < state.energyThreshold) {
                state.terminationStatus = true;
            }

        }
        // Advance time
        state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);

        return state;
    }

protected:
    virtual bool shouldTrain(const StateT& state) const {
        return false;  
    }

    virtual void trainingNeuron(
        const std::unordered_map<std::vector<int>, NeighborData<StateT, double>>& neighborhood,
        StateT& state
    ) const {}

    virtual void updateCellState(
        const std::unordered_map<std::vector<int>, NeighborData<StateT, double>>& neighborhood,
        StateT& state
    ) const {}

    virtual double GetEnergy(
        const std::unordered_map<std::vector<int>, NeighborData<StateT, double>>& neighborhood,
        const StateT& state
    ) const {
        return 0.0; 
    }
};

#endif // NEURON_BASE_GRID_CELL_HPP
