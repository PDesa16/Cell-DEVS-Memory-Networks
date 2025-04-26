#ifndef MODERN_HOPFIELD_HPP
#define MODERN_HOPFIELD_HPP

#include <Eigen/Dense>
#include <cmath>
#include "../BaseModel.hpp"


class ModernHopfield : BaseAssociativeMemoryModel {
    public:
        ModernHopfield(const std::vector<Eigen::VectorXd>& memory_keys, const std::vector<Eigen::VectorXd>& memory_values, double beta = 1.0)
            : beta_(beta) {
            int num = memory_keys.size();
            int dim = memory_keys[0].size();
            keys_ = Eigen::MatrixXd(num, dim);
            values_ = Eigen::MatrixXd(num, memory_values[0].size());
            for (int i = 0; i < num; ++i) {
                keys_.row(i) = memory_keys[i];
                values_.row(i) = memory_values[i];
            }
        }

        // double getKey(int image, int index) {
        //     return keys_(image, index);
        // }

        double getRowSize() {
            return keys_.rows();
        }

        double getKey(int x, int y) const {
            return keys_(x * 50 + y);
        }


        double getValue(int x, int y) const {
            return values_(x,y);
        }

        Eigen::MatrixXd getKeys(const Eigen::VectorXi& indices) const {
            int N = keys_.rows(); 
            int M = indices.size(); 
            Eigen::MatrixXd keys(N, M);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    keys(i, j) = keys_(i, indices(j));
                }
            }
            return keys;
        }

        Eigen::MatrixXd getValues(const Eigen::VectorXi& indices) const {
            int N = values_.rows(); 
            int M = indices.size(); 
            Eigen::MatrixXd values(N, M);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    values(i, j) = values_(i, indices(j));
                }
            }
            return values;
        }

    
        Eigen::VectorXd retrieve(const Eigen::VectorXd& query, const Eigen::VectorXi& indices) const override {
            //getKeys
            auto keys = getKeys(indices);
            auto values = getValues(indices);
            int N = keys_.rows();

            // Step 1: Dot product similarity (query • each key)
            Eigen::VectorXd scores(N);
            for (int i = 0; i < N; ++i) {
                scores(i) = beta_ * query.dot(keys.row(i));
            }
            
            // Step 2: Softmax over scores
            double max_score = scores.maxCoeff();
            Eigen::VectorXd exp_scores = (scores.array() - max_score).exp();
            double sum_exp = exp_scores.sum();
            Eigen::VectorXd weights = exp_scores / sum_exp;
    
            // Step 3: Weighted sum of value vectors
            Eigen::VectorXd output = Eigen::VectorXd::Zero(values.cols());
            for (int i = 0; i < N; ++i) {
                output += weights(i) * values.row(i);
            }
    
            return output;
        }

    
    private:
        Eigen::MatrixXd keys_;    // memory keys (e.g., image vectors)
        Eigen::MatrixXd values_;  // memory values (can be same as keys or labels)
        double beta_;      // attention sharpness
};

#endif