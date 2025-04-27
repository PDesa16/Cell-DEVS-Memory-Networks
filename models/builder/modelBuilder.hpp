#ifndef MODEL_BUILDER_HPP
#define MODEL_BUILDER_HPP

#include <iostream>
#include <variant>
#include <Eigen/Dense>
#include "../../config/simulationConfig.hpp"
#include <cadmium/core/logger/csv.hpp>
#include <cadmium/celldevs/grid/coupled.hpp>
#include <cadmium/core/simulation/root_coordinator.hpp>
#include "../../utils/imageLoader.hpp"
#include "../../utils/generalUtils.hpp"
#include "../../types/imageStructures.hpp"
#include "../cells/states/FieldState.hpp"
#include "../cells/states/HebbState.hpp"
#include "../cells/states/hopfieldState.hpp"
#include "../cells/states/modernHopfieldState.hpp"
#include "../cells/factory/gridCellFactoryBase.hpp"


using namespace cadmium::celldevs;

// --- Global Factory Stuff ---
template<typename StateT>
static std::shared_ptr<GridCellFactoryBase<StateT>> g_factory;


template<typename StateT>
std::shared_ptr<GridCell<StateT, double>> GlobalFactory(
    const coordinates& id,
    const std::shared_ptr<const GridCellConfig<StateT, double>>& config
)
{
    return g_factory<StateT>->create(id, config);
}


enum class SimulationType {
    Hebb,
    Hopfield,
    ModernHopfield,
    Field
};


class ModelBuilder {
    public:
        JSON_LOADER scenarioLoader;
        Scenario scenario;
    
        Eigen::MatrixXd images;       
        Eigen::MatrixXd redImages;
        Eigen::MatrixXd greenImages;
        Eigen::MatrixXd blueImages;
    
        std::variant<
            std::shared_ptr<GridCellDEVSCoupled<FieldState, double>>,
            std::shared_ptr<GridCellDEVSCoupled<HebbState, double>>,
            std::shared_ptr<GridCellDEVSCoupled<HopfieldState, double>>,
            std::shared_ptr<GridCellDEVSCoupled<ModernHopfieldState, double>>
        > neuronCellModel;
    
        std::shared_ptr<cadmium::RootCoordinator> rootCoordinator;
        std::shared_ptr<cadmium::CSVLogger> logger;
    
        ModelBuilder(const std::string& configFile) {
            scenario = scenarioLoader.loadScenario(configFile);
        }
    
        ~ModelBuilder() = default;
    
        ModelBuilder& loadImages() {
            auto imagesRawGray = ImageLoader::LoadImages(scenario.shape[0], scenario.shape[1]);
            int numImages = imagesRawGray.size();
            int numPixels = scenario.shape[0] * scenario.shape[1];
    
            images = Eigen::MatrixXd(numPixels, numImages);
            
            for (int i = 0; i < numImages; ++i) {
                images.col(i) = GeneralUtils::ParseImageToVectorXd(imagesRawGray[i]);
            }
    
            auto imagesRawRGB = ImageLoader::LoadImagesRGB(scenario.shape[0], scenario.shape[1]);
            numImages = imagesRawRGB.size();
    
            redImages = Eigen::MatrixXd(numPixels, numImages);
            greenImages = Eigen::MatrixXd(numPixels, numImages);
            blueImages = Eigen::MatrixXd(numPixels, numImages);
    
            for (int i = 0; i < numImages; ++i) {
                redImages.col(i)   = GeneralUtils::ParseImageToVectorXd(imagesRawRGB[i][0]);
                greenImages.col(i) = GeneralUtils::ParseImageToVectorXd(imagesRawRGB[i][1]);
                blueImages.col(i)  = GeneralUtils::ParseImageToVectorXd(imagesRawRGB[i][2]);
            }
    
            return *this;
        }


    template<typename StateT>
    ModelBuilder& buildNeuronCellModel(
        const std::string& modelName,
        std::shared_ptr<GridCellFactoryBase<StateT>> factory
    )
    {
        g_factory<StateT> = factory;  // Save the passed factory globally
    
        auto model = std::make_shared<GridCellDEVSCoupled<StateT, double>>(
            modelName,
            GlobalFactory<StateT>,    // Pass the static function (plain function pointer)
            "config/simulation_config.json"
        );
    
        model->buildModel();
        neuronCellModel = model;
        return *this;
    }
    

    ModelBuilder& buildLogger(const std::string& logFile) {
        logger = std::make_shared<cadmium::CSVLogger>("logs/" + logFile, ",");
        return *this;
    }

    ModelBuilder& buildRootCoordinator() {
        std::visit([&](auto& model) {
            rootCoordinator = std::make_shared<cadmium::RootCoordinator>(model);
            rootCoordinator->setLogger(logger);
        }, neuronCellModel);
        return *this;
    }


    template<typename StateT, typename CellType = GridCell<StateT, double>>
    Eigen::MatrixXd extractTrainingParameters(int dimension) const {
        Eigen::MatrixXd trainingParameters;
    
        std::visit([&](auto& model) {
            auto components = model->getComponents();
            int numCells = components.size();
            trainingParameters.resize(numCells, dimension); 
    
            int idx = 0;
            for (const auto& [id, component] : components) {
                auto cellGrid = std::dynamic_pointer_cast<CellType>(component);
                if (cellGrid) {
                    auto param = cellGrid->getState().getTrainingParameters();
                    if constexpr (std::is_same_v<decltype(param), double>) {
                        trainingParameters(idx, 0) = param;
                    } else {
                        trainingParameters.row(idx) = param.transpose();
                    }
                    idx++;
                }
            }            
        }, neuronCellModel);
    
        return trainingParameters;
    }

    template<typename StateT, typename CellType = GridCell<StateT, double>>
    Eigen::VectorXd extractActivationStrengths() const {
        Eigen::VectorXd activations;

        std::visit([&](auto& model) {
            auto components = model->getComponents();
            int numCells = components.size();
            activations.resize(numCells);

            int idx = 0;
            for (const auto& [id, component] : components) {
                auto cellGrid = std::dynamic_pointer_cast<CellType>(component);
                if (cellGrid) {
                    activations(idx++) = cellGrid->getState().getActivationStrength();
                }
            }
        }, neuronCellModel);

        return activations;
    }

    SimulationType getSimulationType() const {
        std::string simTypeStr = scenario.simulationType; 
    
        if (simTypeStr == "Hebb") {
            return SimulationType::Hebb;
        } else if (simTypeStr == "Hopfield") {
            return SimulationType::Hopfield;
        } else if (simTypeStr == "ModernHopfield") {
            return SimulationType::ModernHopfield;
        } else if (simTypeStr == "Field") {
            return SimulationType::Field;
        } else {
            throw std::runtime_error("Unknown simulationType: " + simTypeStr);
        }
    }

    int getImageWidth() const {
        return scenario.shape[1]; 
    }
    
    int getImageHeight() const {
        return scenario.shape[0]; 
    }

    template<typename StateT, typename CellType>
    void calculateRMSE(const Eigen::VectorXd& groundTruth, const std::string& simulationName) const {
        Eigen::VectorXd predictions = extractActivationStrengths<StateT, CellType>();
    
        if (groundTruth.size() != predictions.size()) {
            throw std::runtime_error("Ground truth and predictions size mismatch!");
        }
    
        Eigen::VectorXd diff = groundTruth - predictions;
        double mse = diff.squaredNorm() / diff.size();
        double rmse = std::sqrt(mse);
    
        double max_gt = groundTruth.maxCoeff();
        double min_gt = groundTruth.minCoeff();
        double dynamicRange = max_gt - min_gt;
    
        double nrmse = rmse;
        if (dynamicRange > 1e-8) { 
            nrmse = rmse / dynamicRange;
        }
    
        double meanPrediction = predictions.mean();
        double variancePrediction = (predictions.array() - meanPrediction).square().sum() / predictions.size();
        double mae = diff.array().abs().mean();
    
        // Save to log
        std::ofstream logFile("logs/" + simulationName + "_rmse_log.csv", std::ios::app);
        logFile << "RMSE," << rmse << "\n";
        logFile << "nRMSE," << nrmse << "\n"; 
        logFile << "MAE," << mae << "\n";
        logFile << "MeanPrediction," << meanPrediction << "\n";
        logFile << "VariancePrediction," << variancePrediction << "\n";
        logFile.close();
    }
    
    void startSimulation() {
        rootCoordinator->start();
        rootCoordinator->simulate(25.0);
        rootCoordinator->stop();
    }
};

#endif // MODEL_BUILDER_HPP
