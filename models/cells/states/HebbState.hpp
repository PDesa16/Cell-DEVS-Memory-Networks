#ifndef HEBB_STATE_HPP
#define HEBB_STATE_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "../../types/imageStructures.hpp"
#include "../neuronStateInterface.hpp"

struct HebbState : public IGenericState {
    double activationStrength = 0.0;
    double previousActivation = 0.0;
    double energyThreshold = 0.001;
    bool training = true;
    bool terminationStatus = false;
    std::shared_ptr<bool> trainingFlag;
    std::unordered_map<std::vector<int>, double> weights; 
    std::vector<int> coords;

    double getActivationStrength() const { return activationStrength; }
};

#endif // HEBB_STATE_HPP
