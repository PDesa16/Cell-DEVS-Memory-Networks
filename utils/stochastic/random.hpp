#pragma once

#include <random>

// Global random engine
static std::mt19937 gen(std::random_device{}());

class RandomNumberGeneratorDEVS {
public:

    static double generateExponentialDelay(double lambda) {
        std::exponential_distribution<> dis(lambda);
        return dis(gen);
    }

    static double generateGaussianDelay(double mean, double stddev) {
        std::normal_distribution<> gauss_dist(mean, stddev);
        return gauss_dist(gen);
    }

    static double generateUniformDelay(double min, double max) {
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

    static int generateUniformInt(int min, int max) {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

    static void seed(unsigned int seedValue) {
        gen.seed(seedValue);
    }
};
