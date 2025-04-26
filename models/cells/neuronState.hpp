#ifndef NEURON_STATE_HPP
#define NEURON_STATE_HPP

#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include "../../types/imageStructures.hpp"
#include "../../memory_models/modern/ModernHopfield.hpp"
#include <Eigen/Dense>

struct NeuronState {
public:
	double activationStrength;
	double time;
	double prevEnergy = 0;
	int imageWidth;
	int imageHeight;
	std::shared_ptr<ModernHopfield> model;
	Eigen::MatrixXd patternMatrix;
	std::shared_ptr<bool> training = nullptr; 
	bool trainable = true;
	int stabilityCounter = 0;
	NeuronState() = default;
};

// std::ostream& operator<<(std::ostream& os, const NeuronState& s) {
// 	os << std::to_string(s.activationStrength);
// 	return os;
// };

std::ostream& operator<<(std::ostream& os, const NeuronState& s) {

	if (s.training == nullptr){
		return os;
	}

	if (*s.training == true) { 
		// os << std::to_string(s.state(0)) << "," << \
		// std::to_string(s.state(1)) << "," << \
		// std::to_string(s.state(2)) << "," << \
		// std::to_string(s.state(3));
	} else {
		os << std::to_string(s.activationStrength);
	}
	return os;
};


inline bool operator!=(const NeuronState& x, const NeuronState& y) {
	return (x.activationStrength != y.activationStrength) || (x.time != y.time);
};

[[maybe_unused]] void from_json(const nlohmann::json& j, NeuronState& s) {

};


#endif