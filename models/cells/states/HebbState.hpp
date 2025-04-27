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
    int imageWidth;
    int imageHeight;
    bool training = true;
    bool terminationStatus = false;
    std::shared_ptr<bool> trainingFlag;
    std::shared_ptr<std::unordered_map<std::vector<int>, double>> weights; 
    std::vector<int> coords;
    double time = 0.0;
    double learningRate = 0.1;

    double getActivationStrength() const { return activationStrength; }

    double getTrainingParameters() const {
        return weights->at(coords);
    }
};


std::ostream& operator<<(std::ostream& os, const HebbState& s) {
    if (s.training) {
        os << std::to_string(s.weights->at(s.coords));
    } else {
        os << std::to_string(s.getActivationStrength());
    }
    return os;
}


inline bool operator!=(const HebbState& x, const HebbState& y) {
    return (x.getActivationStrength() != y.getActivationStrength()) || (x.time != y.time);
}

[[maybe_unused]] void from_json(const nlohmann::json& j, HebbState& s) {
}


#endif // HEBB_STATE_HPP
