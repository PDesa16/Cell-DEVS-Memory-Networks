#ifndef HOPFIELD_STATE_HPP
#define HOPFIELD_STATE_HPP

#include <memory>
#include <vector>
#include "../../types/imageStructures.hpp"
#include "../neuronStateInterface.hpp"

struct HopfieldState : public IGenericState {
    int activationStatus = 0;
    int previousActivation = 0;
    double energyThreshold = 0.001;
    bool training = false;
    bool terminationStatus = false;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<int> coords;

    std::shared_ptr<WeightMatrix> localWeights;
    std::shared_ptr<StateMatrix> neighboringStates;

    int getActivationStatus() const { return activationStatus; }
};

#endif // HOPFIELD_STATE_HPP
