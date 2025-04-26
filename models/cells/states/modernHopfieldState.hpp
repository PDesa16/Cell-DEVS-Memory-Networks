#ifndef MODERN_HOPFIELD_STATE_HPP
#define MODERN_HOPFIELD_STATE_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "../neuronStateInterface.hpp"

struct ModernHopfieldState : public IGenericState {
    double time = 0.0;
    double activationStrength = 0.0;
    double previousActivation = 0.0;
    double energyThreshold = 0.001;
    bool training = false;
    bool terminationStatus = false;
    Eigen::MatrixXd storedPatterns;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<int> coords;

    double getActivationStrength() const { return activationStrength; }
};

std::ostream& operator<<(std::ostream& os, const ModernHopfieldState& s) {
    if (s.training) {
        os << "Training...";
    } else {
        os << std::to_string(s.getActivationStrength());
    }
    return os;
}

inline bool operator!=(const ModernHopfieldState& x, const ModernHopfieldState& y) {
    return (x.getActivationStrength() != y.getActivationStrength()) || (x.time != y.time);
}

[[maybe_unused]] void from_json(const nlohmann::json& j, ModernHopfieldState& s) {
}

#endif // MODERN_HOPFIELD_STATE_HPP
