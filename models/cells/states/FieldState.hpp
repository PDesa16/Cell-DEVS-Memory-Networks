#ifndef FIELD_STATE_HPP
#define FIELD_STATE_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../../types/imageStructures.hpp"
#include "../neuronStateInterface.hpp"

struct FieldState : public IGenericState {
    Eigen::Vector4d state = Eigen::Vector4d::Zero(); 
    Eigen::VectorXd storedPatterns; 
    bool training = true;
    bool trainable = true;
    int stabilityCounter = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    double time = 0.0;
    double previousActivation = 0.0;
    double energyThreshold = 0.001;
    bool terminationStatus = false;
    std::vector<int> coords;

    double getActivationStrength() const { return state(3); }
    Eigen::Vector4d getTrainingParameters() {
        return state;
    }
};

std::ostream& operator<<(std::ostream& os, const FieldState& s) {
    if (s.training) {
        os << std::to_string(s.state(0)) << ","
           << std::to_string(s.state(1)) << ","
           << std::to_string(s.state(2)) << ","
           << std::to_string(s.state(3));
    } else {
        os << std::to_string(s.getActivationStrength());
    }
    return os;
}

inline bool operator!=(const FieldState& x, const FieldState& y) {
    return (x.getActivationStrength() != y.getActivationStrength()) || (x.time != y.time);
}

[[maybe_unused]] void from_json(const nlohmann::json& j, FieldState& s) {
}

#endif // FIELD_STATE_HPP
