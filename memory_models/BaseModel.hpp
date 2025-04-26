#ifndef BASE_MEMORY_HPP
#define BASE_MEMORY_HPP

#include <Eigen/Dense>
#include <cmath>

class BaseAssociativeMemoryModel {
    public:
        virtual Eigen::VectorXd retrieve(const Eigen::VectorXd& query, const Eigen::VectorXi& indices) const = 0;
};

#endif