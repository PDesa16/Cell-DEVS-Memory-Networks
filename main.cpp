
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "models/builder/modelBuilder.hpp"
#include "utils/stochastic/random.hpp"
#include "utils/generalUtils.hpp"
#include <matplot/matplot.h>
#include "memory_models/modern/ModernHopfield.hpp"
#include "utils/imageLoader.hpp"
#include <cadmium/core/logger/csv.hpp>
#include <cadmium/celldevs/grid/coupled.hpp>
#include <cadmium/core/simulation/root_coordinator.hpp>
#include "models/cells/types/neuronFieldGridCell.hpp"
#include "models/cells/factory/fieldNeuronFactory.hpp"
#include "models/cells/factory/modernHopfieldFactory.hpp"
// 1. Load N grayscale images as memory keys and values (flattened into vectors)
// 2. Load a query image and flatten it
// 3. Compute similarity scores between query and each memory key
// 4. Apply softmax to get attention weights
// 5. Retrieve output vector = weighted sum of memory values
// 6. Reshape and save the output image

std::vector<Eigen::VectorXd> memory_keys_red;
std::vector<Eigen::VectorXd> memory_values_red;
std::vector<Eigen::VectorXd> memory_keys_blue;
std::vector<Eigen::VectorXd> memory_values_blue;
std::vector<Eigen::VectorXd> memory_keys_green;
std::vector<Eigen::VectorXd> memory_values_green;

// Create a Gaussian noise on image to obfuscate recall 
std::shared_ptr<Eigen::VectorXd> gaussianNoiseImage;
std::shared_ptr<bool> training;

void BuildMemoryFromImages(std::vector<std::vector<std::vector<std::vector<uint8_t>>>> images) {

    for (auto& image : images) {

        auto vecRed = GeneralUtils::ParseImageToVectorXd(image[0]);
        auto vecGreen = GeneralUtils::ParseImageToVectorXd(image[1]);
        auto vecBlue = GeneralUtils::ParseImageToVectorXd(image[2]);
        memory_keys_red.push_back(vecRed);
        memory_keys_green.push_back(vecGreen);
        memory_keys_blue.push_back(vecBlue);
    }
    memory_values_red = memory_keys_red;
    memory_values_green  = memory_keys_green;
    memory_values_blue = memory_keys_blue;
}

std::shared_ptr<ModernHopfield> modernHopfieldRed;
std::vector<Eigen::VectorXd> modernHopfieldGreen;
std::shared_ptr<ModernHopfield> modernHopfieldBlue;


// std::shared_ptr<GridCell<NeuronState, double>> addGridCellRed(const coordinates & cellId, const std::shared_ptr<const GridCellConfig<NeuronState, double>>& cellConfig) {
//     // Create a mutable copy of the original config 
//     auto modifiableConfig = std::make_shared<GridCellConfig<NeuronState, double>>(*cellConfig);
//     // double slightPermutation = (*gaussianNoiseImage)(cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);
//     double slightPermutation = modernHopfieldRed->getKey(0, cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);
//     if (modifiableConfig) {
//         modifiableConfig->state.activationStrength =  modernHopfieldRed->getKey(0, cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);
//     }

//     // Adding one pattern (magnitude) to the storedPatterns
//     Eigen::VectorXd storedPatterns(1); 
//     storedPatterns(0) = modernHopfieldRed->getKey(0, cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);  

//     // Example condition: if the cell model is Hebbian-Learning
//     if (cellConfig->cellModel == "Hebbian-Learning") {
//         // Use the modified config to create a new NeuronCell
//         return std::make_shared<NeuronCell>(
//             cellId,                                // std::vector<int>& id
//             modifiableConfig,                       // std::shared_ptr<const GridCellConfig<NeuronState, double>>& config
//             modernHopfieldGreen,                         // std::shared_ptr<BaseAssociativeMemoryModel>& model
//             slightPermutation,                      // double& as
//             modifiableConfig->scenario->shape[0],   // const int& imageWidth
//             modifiableConfig->scenario->shape[1],   // const int& imageHeight
//             0.0,
//             training,                                    // double time
//             storedPatterns
//         );
//     } else {
//         throw std::bad_typeid();
//     }
// }

// std::shared_ptr<GridCell<NeuronState, double>> addGridCellBlue(const coordinates & cellId, const std::shared_ptr<const GridCellConfig<NeuronState, double>>& cellConfig) {
//     // Create a mutable copy of the original config 
//     auto modifiableConfig = std::make_shared<GridCellConfig<NeuronState, double>>(*cellConfig);
//     // double slightPermutation = (*gaussianNoiseImage)(cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);
//     double slightPermutation = modernHopfieldRed->getKey(0, cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);
//     if (modifiableConfig) {
//         modifiableConfig->state.activationStrength =  modernHopfieldBlue->getKey(0, cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);
//     }

//     modifiableConfig->state.state(0) =  RandomNumberGeneratorDEVS::generateUniformDelay(0, 1);
//     modifiableConfig->state.state(1) =  RandomNumberGeneratorDEVS::generateUniformDelay(0, 1);
//     modifiableConfig->state.state(2) =  RandomNumberGeneratorDEVS::generateUniformDelay(0, 1);
//     modifiableConfig->state.state(3) = modifiableConfig->state.activationStrength;

//     // Adding one pattern (magnitude) to the storedPatterns
//     Eigen::VectorXd storedPatterns(1); 
//     storedPatterns(0) = modernHopfieldBlue->getKey(0, cellId[0]*modifiableConfig->scenario->shape[0] + cellId[1]);  

//     // Example condition: if the cell model is Hebbian-Learning
//     if (cellConfig->cellModel == "Hebbian-Learning") {
//         // Use the modified config to create a new NeuronCell
//         return std::make_shared<NeuronCell>(
//             cellId,                                // std::vector<int>& id
//             modifiableConfig,                       // std::shared_ptr<const GridCellConfig<NeuronState, double>>& config
//             modernHopfieldGreen,                         // std::shared_ptr<BaseAssociativeMemoryModel>& model
//             slightPermutation,                      // double& as
//             modifiableConfig->scenario->shape[0],   // const int& imageWidth
//             modifiableConfig->scenario->shape[1],   // const int& imageHeight
//             0.0,
//             training,                                    // double time
//             storedPatterns
//         );
//     } else {
//         throw std::bad_typeid();
//     }
// }



// void calculateWeights(Eigen::MatrixXd& allNeurons, int width, int height) {
//     const double stepSize = 0.001;
//     const double convergenceThreshold = 1e-4;
//     const int maxIterations = 4000;
//     const double epsilon = 1e-6;

//     const int numNeurons = width * height;
//     bool converged = false;
//     int iteration = 0;

//     while (!converged && iteration < maxIterations) {
//         converged = true;
//         iteration++;

//         Eigen::MatrixXd newPositions(3, numNeurons);
//         Eigen::VectorXd newMagnitudes(numNeurons);
//         double maxChange = 0.0;

//         for (int i = 0; i < numNeurons; ++i) {
//             Eigen::Vector3d currentPos = allNeurons.block<3, 1>(0, i);
//             double act_i = allNeurons(3, i);
//             Eigen::Vector3d netForce = Eigen::Vector3d::Zero();
//             double newMag = act_i;

//             for (int j = 0; j < numNeurons; ++j) {
//                 if (i == j) continue;

//                 Eigen::Vector3d neighborPos = allNeurons.block<3, 1>(0, j);
//                 double act_j = allNeurons(3, j);

//                 Eigen::Vector3d delta = currentPos - neighborPos;
//                 double distance = delta.norm();

//                 if (distance > epsilon) {
//                     // Repulsive force to push vectors apart
//                     double strength = (act_i * act_j) / (distance * distance + epsilon);
//                     Eigen::Vector3d force = strength * delta.normalized();
//                     netForce += force;

//                     // Reduce magnitude if dot product is large (not orthogonal)
//                     double alignment = std::abs(currentPos.dot(neighborPos));
//                     newMag -= 0.01 * alignment;
//                 }
//             }

//             Eigen::Vector3d updatedPos = currentPos + stepSize * netForce;
//             if (updatedPos.norm() > epsilon) {
//                 updatedPos.normalize();
//             }

//             newPositions.col(i) = updatedPos;
//             newMagnitudes(i) = std::max(0.0, newMag);

//             double change = (updatedPos - currentPos).norm();
//             if (change > convergenceThreshold) {
//                 converged = false;
//             }
//             maxChange = std::max(maxChange, change);
//         }

//         // Commit updates
//         allNeurons.block(0, 0, 3, numNeurons) = newPositions;
//         allNeurons.row(3) = newMagnitudes.transpose();

//         std::cout << "Iteration: " << iteration << ", Max Change: " << std::fixed << std::setprecision(8) << maxChange << std::endl;

//         if (converged) {
//             std::cout << "Global convergence after " << iteration << " iterations.\n";
//         }
//     }

//     if (!converged) {
//         std::cout << "Maximum iterations reached without convergence.\n";
//     }
// }


int main() {
    // Configuration file path
    std::string configPath = "config/simulation_config.json";

    // Create initial FieldState
    auto initialState = std::make_shared<FieldState>();
    initialState->training = true;
    initialState->imageWidth = 50;
    initialState->imageHeight = 50;

    // Instantiate and build the simulation model
    ModelBuilder builder(configPath);

    builder
        .loadImages();

    // modernHopfieldGreen = builder.greenImages;
    // auto fieldFactory = std::make_shared<FieldNeuronFactory>(builder.greenImages);
    auto modernFactoryRecall = std::make_shared<ModernHopfieldNeuronFactory>(builder.greenImages, \
        GeneralUtils::GaussianBlurFlatImages(builder.greenImages,50,50, 5, 9), \
        true);
    
    builder
        .buildNeuronCellModel<ModernHopfieldState>("Modern Hopfield Network", modernFactoryRecall)
        .buildLogger("modern_hopfield_model_log.csv")
        .buildRootCoordinator()
        .startSimulation();

    return 0;
}
// .buildNeuronCellModel<FieldState>("Field-VectorModel", fieldFactory)

// int main() {
//     training = std::make_shared<bool>(true);

//     auto images = ImageLoader::LoadImagesRGB(50, 50);
//     BuildMemoryFromImages(images);
//     modernHopfieldRed = std::make_shared<ModernHopfield>(memory_keys_red, memory_values_red, /*beta=*/2.0);
//     modernHopfieldBlue = std::make_shared<ModernHopfield>(memory_keys_blue, memory_values_blue, /*beta=*/2.0);
//     modernHopfieldGreen = std::make_shared<ModernHopfield>(memory_keys_green, memory_values_green, /*beta=*/2.0);


//     Eigen::VectorXd blurImage = GeneralUtils::GaussianBlurFlatImage(memory_keys_green[0],50,50, 5, 9);
//     gaussianNoiseImage =  std::make_shared<Eigen::VectorXd>(blurImage);
//     // Eigen::MatrixXd matOutput = Eigen::Map<Eigen::MatrixXd>(test.data(), 50, 50);
//     // std::vector<std::vector<double>> Z(50, std::vector<double>(50));

//     // for (int i = 0; i < 50; ++i)
//     //     for (int j = 0; j < 50; ++j)
//     //         Z[i][j] = matOutput(i, j);

//     // matplot::image(Z);
//     // matplot::show();

//     std::shared_ptr<GridCellDEVSCoupled<NeuronState, double>> neuronCellModel;
//     std::shared_ptr<cadmium::RootCoordinator> rootCoordinator;

//     neuronCellModel = std::make_shared<GridCellDEVSCoupled<NeuronState, double>>(
//         "Hebbian-Learning",
//         addGridCellGreen,
//         "simulation_config.json"
//     );

//     neuronCellModel->buildModel();


//     // Learning Energy Landscape
//     auto loggerTrain = std::make_shared<cadmium::CSVLogger>("logs/training.csv", ",");
//     rootCoordinator = std::make_shared<cadmium::RootCoordinator>(neuronCellModel); 
//     rootCoordinator -> setLogger(loggerTrain);
//     rootCoordinator -> start();
//     rootCoordinator -> simulate(25.0);
//     rootCoordinator -> stop();


//     // Set Magnitudes = Gaussian image
//     // auto comps1 = neuronCellModel->getComponents();
//     // Eigen::MatrixXd weightMatrix(4, 50 * 50);
//     // int i = 0;
//     // for (auto& comp : comps1) {
//     //     auto cellGrid = std::dynamic_pointer_cast<NeuronCell>(comp.second);
//     //     weightMatrix.col(i) = cellGrid->getState().state; // Make sure state is Eigen::VectorXd with 4 elements
//     //     ++i;
//     // }


//     std::vector<Eigen::MatrixXd> fullWeightSet;  // size: numNeurons, each entry is [N x 4]
//     auto comps1 = neuronCellModel->getComponents();
//     for (const auto& [_, comp] : comps1) {
//         auto cellGrid = std::dynamic_pointer_cast<NeuronCell>(comp);
//         fullWeightSet.push_back(cellGrid->getPatternMatrix());  // assume getPatternMatrix() returns [N x 4]
//     }


//     // calculateWeights(weightMatrix, 50, 50);

//     // Inference 

//     // // Set Magnitudes = Gaussian image
//     // auto comps = neuronCellModel->getComponents();
//     // for (auto comp : comps) {
//     //     auto cellGrid = std::dynamic_pointer_cast<NeuronCell>(comp.second);
//     //     auto [x, y] = GeneralUtils::stringToIndices(comp.first);
//     //     // state.state(3) = (*gaussianNoiseImage)(x*50 + y) ; 
//     //     cellGrid->setWeight(0.1);
//     // }

//     std::shared_ptr<cadmium::RootCoordinator> rootCoordinator2;
//     auto neuronCellModel2 = std::make_shared<GridCellDEVSCoupled<NeuronState, double>>(
//         "Hebbian-Learning",
//         addGridCellGreen,
//         "simulation_config.json"
//     );

//     neuronCellModel2->buildModel();
//     rootCoordinator2 = std::make_shared<cadmium::RootCoordinator>(neuronCellModel2); 


//     // // Set Magnitudes = Gaussian image
//     // auto comps = neuronCellModel2->getComponents();
//     // int j = 0;
//     // for (auto& comp : comps) {
//     //     auto cellGrid = std::dynamic_pointer_cast<NeuronCell>(comp.second);
//     //     cellGrid->setWeights(weightMatrix.col(j)); // Make sure state is Eigen::VectorXd with 4 elements
//     //     auto [x, y] = GeneralUtils::stringToIndices(comp.first);
//     //     cellGrid->setWeight((*gaussianNoiseImage)(x*50 + y));
//     //     // cellGrid->setWeight(0.1);
//     //     ++j;
//     // }

//     // Set pattern matrix and activation strength per neuron
//     auto comps = neuronCellModel2->getComponents();
//     int j = 0;
//     for (auto& comp : comps) {
//         auto cellGrid = std::dynamic_pointer_cast<NeuronCell>(comp.second);
//         auto [x, y] = GeneralUtils::stringToIndices(comp.first);

//         // Extract N x 4 pattern matrix for this neuron from full weight matrix
//         Eigen::MatrixXd patternBlock(2, 4);
//         for (int p = 0; p < 2; ++p) {
//             patternBlock = fullWeightSet[j];  // assumes weightMatrix has shape [4 x (numNeurons * numPatterns)]
//         }

//         cellGrid->setWeights(patternBlock);  // now stores all N patterns

//         // Set global activation strength (e.g., Gaussian image per neuron)
//         cellGrid->setWeight((*gaussianNoiseImage)(x * 50 + y));
//         ++j;
//     }



//     *training = false;
//     auto loggerRecall = std::make_shared<cadmium::CSVLogger>("logs/recall.csv", ",");
//     rootCoordinator2 -> setLogger(loggerRecall);
//     rootCoordinator2 -> start();
//     rootCoordinator2 -> simulate(25.0);

//     // Store Learned weights
//     // Use learned weights in recall simulation




//     // Run sim
// };






// int main() {

    // std::vector<Eigen::VectorXd> memory_keys;
    // std::vector<Eigen::VectorXd> memory_values;
    // auto images = ImageLoader::LoadImages(100,100);

    // for (auto& image : images) {
    //     auto vec = GeneralUtils::ParseImageToVectorXd(image);
    //     std::cout << vec << std::endl;
    //     memory_keys.push_back(vec);
    //     std::cout << "test" << std::endl;
    // }
    
    // // In simplest case, keys == values
    // memory_values = memory_keys;
    
    // ModernHopfield hopfield(memory_keys, memory_values, /*beta=*/2.0);
    
    // Eigen::VectorXd query = memory_keys[0];

    // for (int i=0; i< query.size(); i++) {
    //     if (i % 5 == 0) {
    //         if (query(i) > 0.5) {
    //             query(i) = 0;
    //         } else {
    //             query(i) = 1;
    //         }
        
    //     }
    // }
    // std::vector<std::vector<double>> L(100, std::vector<double>(100));
    // Eigen::MatrixXd matQuery = Eigen::Map<Eigen::MatrixXd>(query.data(), 100, 100); // 2 rows, 3 cols
    // for (int i = 0; i < 100; ++i)
    //     for (int j = 0; j < 100; ++j)
    //         L[i][j] = matQuery(i, j);

    // matplot::image(L);
    // matplot::show();

    // // std::cout << matQuery << std::endl;
    // Eigen::VectorXd output = hopfield.retrieve(query);
    // Eigen::MatrixXd matOutput = Eigen::Map<Eigen::MatrixXd>(output.data(), 100, 100); // 2 rows, 3 cols
    // // std::cout << matOutput << std::endl;

    // std::vector<std::vector<double>> Z(100, std::vector<double>(100));

    // for (int i = 0; i < 100; ++i)
    //     for (int j = 0; j < 100; ++j)
    //         Z[i][j] = matOutput(i, j);

    // matplot::image(Z);
    // matplot::show();

// }

// int main() {
//     // Define memory keys and values
//     MatrixXd keys(3, 4);   // 3 memories, 4D keys
//     MatrixXd values(3, 2); // 3 values, 2D output vectors

//     keys << 1, 0, 0, 0,
//             0, 1, 0, 0,
//             0, 0, 1, 0;

//     values << 1, 0,
//               0, 1,
//               1, 1;

//     // Define a query vector (could be noisy or partial)
//     VectorXd query(4);
//     query << 0.9, 0.1, 0, 0;

//     // Retrieve associated value
//     VectorXd result = hopfield_retrieve(query, keys, values);

//     cout << "Retrieved vector:\n" << result << endl;

//     return 0;
// }
