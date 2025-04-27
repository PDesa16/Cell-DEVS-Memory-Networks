#ifndef HOPFIELD_STATE_HPP
#define HOPFIELD_STATE_HPP

#include <memory>
#include <vector>
#include "../../types/imageStructures.hpp"
#include "../neuronStateInterface.hpp"

struct HopfieldState : public IGenericState {
    double activationStrength = 0.0;
    double energyThreshold = 0.001;
    bool training = false;
    bool terminationStatus = false;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<int> coords;
    double time = 0.0;

    const Eigen::MatrixXd* weights = nullptr; 

    double getActivationStrength() const { return activationStrength; }
};

std::ostream& operator<<(std::ostream& os, const HopfieldState& s) {
    if (s.training) {
        os << "Training...";
    } else {
        os << std::to_string(s.getActivationStrength());
    }
    return os;
}

inline bool operator!=(const HopfieldState& x, const HopfieldState& y) {
    return (x.getActivationStrength() != y.getActivationStrength()) || (x.time != y.time);
}

[[maybe_unused]] void from_json(const nlohmann::json& j, HopfieldState& s) {
}

#endif // HOPFIELD_STATE_HPP
