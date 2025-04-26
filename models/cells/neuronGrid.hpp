#ifndef NEURON_CELL_HPP
#define NEURON_CELL_HPP

#include <cmath>
#include "neuronState.hpp"
#include "../../types/imageStructures.hpp"
#include "../../utils/generalUtils.hpp"
#include <cadmium/celldevs/grid/cell.hpp>
#include <cadmium/celldevs/grid/config.hpp>
#include "../../utils/stochastic/random.hpp"
#include <cstdlib> 

using namespace cadmium::celldevs;

class NeuronCell : public GridCell<NeuronState, double> {
public:
    // Constructor
    NeuronCell(const std::vector<int>& id,
        const std::shared_ptr<const cadmium::celldevs::GridCellConfig<NeuronState, double>>& config,
        std::shared_ptr<ModernHopfield> model,  // Removed reference here
        double as, 
        const int& imageWidth,
        const int& imageHeight,
        double time,
        std::shared_ptr<bool> training, 
        Eigen::MatrixXd patternMatrix): GridCell<NeuronState, double>(id, config) {
                this->state.model = model;
                this->state.activationStrength = as;
                this->state.imageWidth = imageWidth;
                this->state.imageHeight = imageHeight;
                this->state.time = time;
                this->state.training = training;
                this->state.patternMatrix = patternMatrix;
            }

    // Default added for testing
    NeuronCell() = default;
            
            // [[nodiscard]] NeuronState localComputation(NeuronState state,
            //     const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood) const {
            //     if (state.time != 0) {
            //         // Get current coords
            //         auto [x, y] = GeneralUtils::stringToIndices(this->getId());
            //         // Update state
            //         updateCellState(x,y,neighborhood,state);
            //     }
            //     // Advance time
            //     state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);
            //     // Return current state which is passed to neighbors ports
            //     return state;
            // }


            static double sigmoid(double x) {
                return 1.0 / (1.0 + std::exp(-x));
            }

            static double sigmoid_derivative(double x) {
                return sigmoid(x) / (1.0 - sigmoid(x));
            }
            
            // // state.synapticWeights
            // [[nodiscard]] NeuronState localComputation(
            //     NeuronState state,
            //     const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood) const {
                
            //     if (state.time != 0) {
            //         auto [x, y] = GeneralUtils::stringToIndices(this->getId());
            
            //         // --- Step 1: Compute current output (weighted sum of neighbor activations) ---
            //         double weightedSum = 0.0;
            //         for (const auto& [coord, neighbor] : neighborhood) {
            //             double input = neighbor.state->activationStrength;
            //             double weight = state.model->getKey(coord[0],coord[1]);
            //             weightedSum += weight * input;
            //         }
            
            //         // Optional: Activation function (sigmoid, tanh, ReLU, etc.)
            //         double out = NeuronCell::sigmoid(weightedSum);
            
            //         // --- Step 2: Update weights using Oja’s Rule (only during training) ---
            //         // if (state.training) {
            //             const double eta = 0.01;  // learning rate
            //             for (const auto& [coord, neighbor] : neighborhood) {
            //                 double x_i = neighbor.state->activationStrength;
            //                 double  w_i = state.model->getKey(coord[0],coord[1]); // ref to modify directly
            //                 w_i += eta * out * (x_i - out * w_i);
            //             }
            //         // }
            
            //         // --- Step 3: Set new activation ---
            //         state.activationStrength = out;
            //     }
            
            //     // Step 4: Advance time
            //     state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);
            
            //     return state;
            // }


            // [[nodiscard]] NeuronState localComputation(
            //     NeuronState state,
            //     const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood
            // ) const {

            //     // if (*state.training == false && state.time < 150) {
            //     //     clearedNeighborhood.clear();
            //     //     // Use clearedNeighborhood instead of neighborhood for inference
            //     // }
                

            //     if (state.time != 0) {
            //         Eigen::Vector3d currentPos = state.state.head<3>();
            //         double currentMag = state.state(3);
            //         double act_i = currentMag;
            //         Eigen::Vector3d netForce = Eigen::Vector3d::Zero(); // Only used in train mode
            //         double sumHebb = 0.0;
            
            //         for (const auto& [coord, neighbor] : neighborhood) {
            //             Eigen::Vector3d neighborPos = neighbor.state->state.head<3>();
            //             double neighborMag = neighbor.state->state(3);
            //             double act_j = neighborMag;
            //             double alignment = currentPos.normalized().dot(neighborPos.normalized());
            
            //             Eigen::Vector3d delta = neighborPos - currentPos;
            //             double distance = delta.norm();
            
            //             if (distance > 1e-6) {
            //                 // Force computation based on activations and inverse square distance
            //                 double strength = (act_i * act_j) / ((distance * distance) + 1e-6);
            //                 Eigen::Vector3d force = strength * delta.normalized();
            
            //                 if (*state.training == true) {
            //                     netForce += force;
            //                 } else {
            //                     sumHebb += act_i * act_j * alignment;  // Hebbian term for recall
            //                     // * std::exp(-1 * (distance*distance) / (4.0));
            //                     // act_i * act_j How much should j influence i determined by its projection on i 
            //                 }
            //             }
            //         }
            
            //         // Apply updates depending on mode
            //         if (*state.training == true) {
            //             // Move position based on total force
            //             Eigen::Vector3d updatedPos = currentPos + 0.1 * netForce;
            //             if (updatedPos.norm() > 1e-6) {
            //                 updatedPos.normalize();  // make it unit length
            //             }
            //             state.state.head<3>() = updatedPos;
            //             // if (this->getId() == "(46,34)") {
            //             //     std::cout << state.state.head<3>() << "Neuron: " <<  this->getId() << std::endl;
            //             // }
            //         } else {
            //             // Update magnitude only (positions fixed)
            //             const double eta = 0.1;
            //             const double lambda = 0.01;
            //             double deltaM = eta * std::tanh(sumHebb);
            //             state.state(3) = std::clamp(currentMag + deltaM, 0.0, 1.0);
            //         }
            //     }
            //     state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);
            //     return state;
            // }
            




            // [[nodiscard]] NeuronState localComputation(
            //     NeuronState state,
            //     const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood
            // ) const {
            //     if (state.time != 0) {
            //         Eigen::Vector3d currentPos = state.state.head<3>();
            //         double currentMag = state.state(3);
            //         Eigen::Vector3d netForce = Eigen::Vector3d::Zero();
            //         double sumHebb = 0.0;
            //         double denominator = 0.0;
            
            //         for (const auto& [coord, neighbor] : neighborhood) {
            //             Eigen::Vector3d neighborPos = neighbor.state->state.head<3>();
            //             double neighborMag = neighbor.state->state(3);
            //             Eigen::Vector3d delta = neighborPos - currentPos;
            //             double distance = delta.norm();
            
            //             if (distance > 1e-6) {
            //                 double invDistSq = 1.0 / (distance * distance + 1e-6);
            //                 Eigen::Vector3d force = -currentMag * neighborMag * invDistSq * delta.normalized();
            //                 netForce += force;
            //                 denominator += neighborMag * invDistSq;
            
            //                 // For Hebbian term
            //                 double alignment = currentPos.normalized().dot(neighborPos.normalized());
            //                 sumHebb += currentMag * neighborMag * alignment;
            //             }
            //         }
            
            //         if (*state.training == true && state.trainable) {
            //             // TRAINING: update position using net force
            //             Eigen::Vector3d updatedPos = currentPos + 0.1 * netForce;
            //             if (updatedPos.norm() > 1e-6) {
            //                 updatedPos.normalize();
            //             }
            
            //             double movement = (updatedPos - currentPos).norm();
            //             state.state.head<3>() = updatedPos;
            
            //             // Stability tracking
            //             const double epsilon = 1e-4;
            //             state.stabilityCounter = (movement < epsilon) ? state.stabilityCounter + 1 : 0;
            
            //             const int stabilityThreshold = 5;
            //             if (state.stabilityCounter >= stabilityThreshold) {
            //                 state.trainable = false;
            //             }
            //         } else {
            //             Eigen::Vector3d vi = currentPos.normalized();
            //             double weightedSum = 0.0;

            //             for (const auto& [coord, neighbor] : neighborhood) {
            //                 Eigen::Vector3d delta = neighbor.state->state.head<3>() - currentPos;
            //                 double distance = delta.norm();
            //                 if (distance > 1e-6) {
            //                     // Eigen::Vector3d unitDelta = delta.normalized();
            //                     // double alignment = unitDelta.dot(vi);
            //                     Eigen::Vector3d vj = neighbor.state->state.head<3>().normalized();
            //                     double alignment = -1 * vi.dot(vj);
            //                     double mag_j = neighbor.state->state(3);
            //                     // double diff = currentMag - mag_j;
            //                     // double diff = currentMag - mag_j;
            //                     // double pushPull = std::tanh(diff);
            //                     // weightedSum += pushPull * mag_j * alignment * -1;
            //                     double magDiff = currentMag - mag_j;
            //                     double influence = std::tanh(magDiff * alignment); // smoother interpolation
            //                     weightedSum += influence;
            //                 }
            //             }

            //             // Leak decay
            //             double decay = 0.01;
            //             weightedSum -= decay * currentMag;                   
            //             // Update
            //             const double alpha = 0.1; // could be <1.0 if smoother updates are desired
            //             state.state(3) = std::clamp(currentMag + alpha * weightedSum, 0.0, 1.0);

            //         }
            //     }
            //     state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);
            //     return state;
            // }
            
            // NeuronState localComputation(
            //     NeuronState state,
            //     const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood
            // ) const {
            //     if (state.time != 0) {
            //         Eigen::Vector3d currentPos = state.state.head<3>();
            //         double currentMag = state.state(3);
            //         Eigen::Vector3d netForce = Eigen::Vector3d::Zero();
            //         double sumHebb = 0.0;
            //         double denominator = 0.0;
            
            //         // Local interactions
            //         for (size_t i = 0; i < state.storedPatterns.size(); ++i) {
            //             for (const auto& [coord, neighbor] : neighborhood) {
            //                 Eigen::Vector3d neighborPos = neighbor.state->state.head<3>();
            //                 double neighborMag = neighbor.state->state(3);
            //                 Eigen::Vector3d delta = neighborPos - currentPos;
            //                 double distance = delta.norm();
                
            //                 if (distance > 1e-6) {
            //                     double invDistSq = 1.0 / (distance * distance + 1e-6);
            //                     Eigen::Vector3d unitDelta = delta.normalized();
            //                     double alignment = currentPos.normalized().dot(neighborPos.normalized());
                
            //                     Eigen::Vector3d force = -currentMag * neighborMag * invDistSq * unitDelta;
            //                     netForce += alignment * force;
            //                 }
            //             }
            //             netForce /= state.storedPatterns.size();
            //         }
            
            //         // --- UPDATE POSITION (TRAINING) ---
            //         if (*state.training == true && state.trainable) {
            //             Eigen::Vector3d updatedPos = currentPos + 0.1 * netForce;
            //             if (updatedPos.norm() > 1e-6) {
            //                 updatedPos.normalize();
            //             }
            
            //             double movement = (updatedPos - currentPos).norm();
            //             state.state.head<3>() = updatedPos;
            
            //             // Stability tracking
            //             const double epsilon = 1e-4;
            //             state.stabilityCounter = (movement < epsilon) ? state.stabilityCounter + 1 : 0;
            
            //             const int stabilityThreshold = 5;
            //             if (state.stabilityCounter >= stabilityThreshold) {
            //                 state.trainable = false;
            //             }
            
            //         } else {


            //         Eigen::Vector3d vi = currentPos.normalized();
            //         double weightedSum = 0.0;

            //         for (const auto& [coord, neighbor] : neighborhood) {
            //             Eigen::Vector3d delta = neighbor.state->state.head<3>() - currentPos;
            //             double distance = delta.norm();
            //             if (distance > 1e-6) {
            //                 Eigen::Vector3d vj = neighbor.state->state.head<3>().normalized();
            //                 double alignment = -1 * vi.dot(vj);
            //                 double mag_j = neighbor.state->state(3);
            //                 double magDiff = currentMag - mag_j;
            //                 double influence = std::tanh(magDiff * alignment); // smoother interpolation
            //                 weightedSum += influence;
            //             }
            //         }
            //         // --- GLOBAL RECALL (Modern Hopfield) ---
            //         double beta = 10.0;  // sharpness of recall
            //         double total = 0.0;
            //         std::vector<double> weights;

            //         for (const auto& magnitude : state.storedPatterns) {
            //             // Compare based on magnitude similarity
            //             double sim = std::abs(currentMag - magnitude);  // Difference in magnitudes
            //             double score = std::exp(-beta * sim);  // Negative because smaller magnitude difference is more similar
            //             weights.push_back(score);
            //             total += score;
            //         }

            //         // Compute weighted average of stored magnitudes for recall
            //         double globalRecall = 0.0;
            //         for (size_t i = 0; i < state.storedPatterns.size(); ++i) {
            //             globalRecall += (weights[i] / total) * state.storedPatterns[i];
            //         }

            //         // Update magnitude
            //         const double alpha = 0.1;  // Update smoothing factor
            //         double combinedUpdate = currentMag + alpha * weightedSum;  // Local force-based update
            //         state.state(3) = std::clamp(0.5 * combinedUpdate + 0.5 * globalRecall, 0.0, 1.0);  // Blend local and global updates
            //         }
            //     }
            
            //     state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);
            //     return state;
            // }


            NeuronState localComputation(
                NeuronState state,
                const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood
            ) const {
                if (state.time != 0) {
                    int numPatterns = state.patternMatrix.rows();
            
                    if (*state.training == true && state.trainable) {
                        const double epsilon = 1e-6;
            
                        for (int p = 0; p < numPatterns; ++p) {
                            Eigen::Vector3d currentPos = state.patternMatrix.block<1, 3>(p, 0).transpose();
                            double currentMag = state.patternMatrix(p, 3);
                            Eigen::Vector3d netForce = Eigen::Vector3d::Zero();
            
                            for (const auto& [coord, neighbor] : neighborhood) {
                                Eigen::Vector3d neighborPos = neighbor.state->patternMatrix.block<1, 3>(p, 0).transpose();
                                double neighborMag = neighbor.state->patternMatrix(p, 3);
                                Eigen::Vector3d delta = neighborPos - currentPos;
                                double distance = delta.norm();
            
                                if (distance > epsilon) {
                                    double invDistSq = 1.0 / (distance * distance + epsilon);
                                    Eigen::Vector3d unitDelta = delta.normalized();
                                    double alignment = currentPos.normalized().dot(neighborPos.normalized());
                                    Eigen::Vector3d force = -currentMag * neighborMag * invDistSq * unitDelta;
                                    netForce += alignment * force;
                                }
                            }
            
                            Eigen::Vector3d updatedPos = currentPos + 0.1 * netForce;
                            if (updatedPos.norm() > epsilon) {
                                updatedPos.normalize();
                            }
            
                            double movement = (updatedPos - currentPos).norm();
                            state.patternMatrix.block<1, 3>(p, 0) = updatedPos.transpose();
            
                            // Optional: Update magnitude if you're doing pattern-wise dynamics
                            state.patternMatrix(p, 3) = std::max(0.0, currentMag); // Can refine
            
                            if (movement > 1e-4) {
                                state.stabilityCounter = 0;
                            } else {
                                state.stabilityCounter++;
                            }
            
                            if (state.stabilityCounter >= 5) {
                                state.trainable = false;
                            }
                        }
            
                    } else {
                    // --- RECALL ONLY ---
                    double beta = 10.0;
                    double maxTotalScore = -1.0;
                    double bestActivation = state.activationStrength;  // Default to current value

                    for (int p = 0; p < numPatterns; ++p) {
                        Eigen::Vector3d currentPos = state.patternMatrix.block<1, 3>(p, 0).transpose();
                        double currentMag = state.activationStrength;
                        Eigen::Vector3d vi = currentPos.normalized();

                        double weightedSum = 0.0;

                        // Local update for this pattern
                        for (const auto& [coord, neighbor] : neighborhood) {
                            Eigen::Vector3d vj = neighbor.state->patternMatrix.block<1, 3>(p, 0).transpose().normalized();
                            double mag_j = neighbor.state->patternMatrix(p, 3);
                            double magDiff = currentMag - mag_j;
                            double influence = std::tanh(magDiff * (-vi.dot(vj)));
                            weightedSum += influence;
                        }

                        double combinedUpdate = currentMag + 0.1 * weightedSum;

                        // Global recall: compare combinedUpdate to all stored patterns
                        double totalScore = 0.0;
                        for (int k = 0; k < numPatterns; ++k) {
                            double sim = std::abs(combinedUpdate - state.patternMatrix(k, 3));
                            double score = std::exp(-beta * sim);
                            totalScore += score;
                        }

                        // If this pattern gave us the highest matching score, save it
                        if (totalScore > maxTotalScore) {
                            maxTotalScore = totalScore;
                            bestActivation = combinedUpdate;
                        }
                    }

                    // Update relative to the most confident pattern
                    state.activationStrength = std::clamp(0.5*bestActivation + 0.5*maxTotalScore, 0.0, 1.0);

                    }
                }
            
                state.time += RandomNumberGeneratorDEVS::generateExponentialDelay(1);
                return state;
            }            
            

            
            
            
            


            
            
            // bool verifyImmidiateNeighborsHaveFired(int x, int y, const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood, NeuronState& state) const {
            //     int WIDTH_EDGE = state.imageWidth - 1;
            //     int LENGTH_EDGE = state.imageLength - 1;
            //     // Right Nearest 
            //     auto rightNearest = (x < WIDTH_EDGE)  ?  GeneralUtils::parseCellIndexToCadmiumId(x + 1, y) : "";
            //     // Left Nearest 
            //     auto leftNearest = (x > 0) ? GeneralUtils::parseCellIndexToCadmiumId(x - 1, y) : "";
            //     // Top Nearest 
            //     auto topNearest = (y > 0) ? GeneralUtils::parseCellIndexToCadmiumId(x , y - 1) : "";
            //     // Bottom Nearest 
            //     auto bottomNearest = (y < LENGTH_EDGE) ? GeneralUtils::parseCellIndexToCadmiumId(x , y + 1) : "";
            
            //     // Check if any of our immidiate neighbors are active.
            //     bool isActive = false;
            //     for (const auto& [neighborId, neighborData] : neighborhood) {
            //         int neighborState = neighborData.state->activationStatus;
            //         auto neighborStringID = GeneralUtils::parseCellIndexToCadmiumId(neighborId[0], neighborId[1]);
            //         if (neighborState == 1 && 
            //             (neighborStringID == rightNearest || neighborStringID == leftNearest || 
            //             neighborStringID == topNearest || neighborStringID == bottomNearest)) {
            //             isActive = true;
            //             break;
            //         }
            //     }
            
            //     return isActive;
            // };

            // bool neighborhoodIsActive(const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood) const {
            //     bool isActive = false;
            //     for (const auto& [neighborId, neighborData] : neighborhood) {
            //         int neighborState = neighborData.state->activationStatus;
            //         if (neighborState == 1) {
            //             isActive = true;
            //             break;
            //         }
            //     }
            //     return isActive;
            // }
            
            
            void updateCellState(int x, int y ,const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood, NeuronState& state) const {
                // NxN matrix 
                // i.e row 0 represents the product of cell 0 with every other cell including itself. 
                // Each column represents the other cell i.e col 1 = cell0 * cell1, col 2 = cell0 * cell2 ... etc
                int neighborhoodSize = neighborhood.size() + 1;  // +1 for self
                int selfFlatIndex = x * state.imageWidth + y;  // Assuming x and y are the coordinates of the cell
                Eigen::VectorXi indices(neighborhoodSize);  // Create the vector of indices
                Eigen::VectorXd states(neighborhoodSize);  // Create the vector of states
                
                // Initialize the first element with the self index and self state
                indices(0) = selfFlatIndex;
                states(0) = state.activationStrength;
                
                // Iterate through neighboring neurons
                int idx = 1;  // Start from the second element (after 'self')
                for (const auto& [neighborId, neighborData] : neighborhood) {
                    indices(idx) = neighborId[0] *  state.imageWidth + neighborId[1];  // Set the neighbor index
                    states(idx) = neighborData.state->activationStrength;  // Set the neighbor state
                    ++idx;  // Move to the next index in the vectors
                }
                auto vecRet = state.model->retrieve(states, indices);

                state.activationStrength = vecRet[0];

            }


            // void updateCellState(int x, int y ,const std::unordered_map<std::vector<int>, NeighborData<NeuronState, double>>& neighborhood, NeuronState& state) const {
            //     // NxN matrix 
            //     // i.e row 0 represents the product of cell 0 with every other cell including itself. 
            //     // Each column represents the other cell i.e col 1 = cell0 * cell1, col 2 = cell0 * cell2 ... etc
            //     double sum = 0;
            //     int selfFlatIndex = x * state.imageWidth + y;
            //     // Iterate through neighboring neurons
            //     for (const auto& [neighborId, neighborData] : neighborhood) {
            //         int neighborState = neighborData.state->activationStatus;
            
            //         auto neighborStringID = GeneralUtils::parseCellIndexToCadmiumId(neighborId[0], neighborId[1]);
            //         int neighborFlatIndex = neighborId[0] * state.imageWidth + neighborId[1];
            
            //         // Perform update rule
            //         if (neighborStringID != this->getId()) {
            //             sum += state.localWeights->getWeightAt(selfFlatIndex, neighborFlatIndex) * static_cast<double>(neighborState);
            //         }
            //     }
            //     // Update activation state using the sum
            //     state.activationStatus = GeneralUtils::signum(sum);
            //     // Set it in the global state matrix
            //     state.neighboringStates->stateMatrix(x,y) = state.activationStatus;
            // }
            
            [[nodiscard]] double outputDelay(const NeuronState& state) const {
                return state.time;
            }
            
            NeuronState& getState(){ return this->state; }

        //    void setWeight(double weight) { this->state.state(3) = weight; }
        void setWeight(double weight) { this->state.activationStrength = weight; }

        // // void setWeights(Eigen::VectorXd weights) {this->state.state = weights;}
        // void setWeights(int i, const Eigen::Vector4d& weights) {
        //     this->state.patternMatrix.row(i) = weights.transpose();  // Set the i-th pattern [x, y, z, m]
        // }

        // void setWeights(Eigen::VectorXd weights) {this->state.state = weights;}
        void setWeights(const Eigen::MatrixXd& weights) {
            this->state.patternMatrix = weights;  // Set the i-th pattern [x, y, z, m]
        }

        Eigen::MatrixXd& getPatternMatrix() {
            return this->state.patternMatrix;
        }
        
        
           
        
};

#endif
