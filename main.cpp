
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "models/builder/modelBuilder.hpp"
#include "utils/stochastic/random.hpp"
#include "utils/generalUtils.hpp"
#include <matplot/matplot.h>
#include "utils/imageLoader.hpp"
#include <cadmium/core/logger/csv.hpp>
#include <cadmium/celldevs/grid/coupled.hpp>
#include <cadmium/core/simulation/root_coordinator.hpp>
#include "models/cells/types/neuronFieldGridCell.hpp"
#include "models/cells/factory/fieldNeuronFactory.hpp"
#include "models/cells/factory/modernHopfieldFactory.hpp"
#include "models/cells/factory/hopfieldFactory.hpp"
#include "models/cells/factory/hebbFactory.hpp"

int main() {
    // Configuration file path
    std::string configPath = "config/simulation_config.json";

    // Instantiate and build the simulation model
    ModelBuilder builder(configPath);

    builder.loadImages();

    // Automatically detect simulation type
    SimulationType simType = builder.getSimulationType();

    // Get image size
    int width = builder.getImageWidth();
    int height = builder.getImageHeight();

    // Select image as the ground truth
    Eigen::VectorXd groundTruth = builder.greenImages.col(0);

    switch (simType) {
    case SimulationType::Hebb: {
        // Training
        auto hebbFactoryTraining = std::make_shared<HebbNeuronFactory>(
            builder.greenImages,
            GeneralUtils::GaussianBlurFlatImagesWithPermutation(builder.greenImages, width, height, 5, 9, 0.5)
        );

        builder
        .buildNeuronCellModel<HebbState>("Hebb Network", hebbFactoryTraining)
        .buildLogger("training_hebb_model_log.csv")
        .buildRootCoordinator()
        .startSimulation();

        // Get the training weights
        auto trainingVector = builder.extractTrainingParameters<HebbState, NeuronHebbGridCell>(1);

        // Recall
        auto hebbFactoryRecall = std::make_shared<HebbNeuronFactory>(
            builder.greenImages,
            GeneralUtils::GaussianBlurFlatImagesWithPermutation(builder.greenImages, width, height, 5, 9, 0.5),
            trainingVector
        );        

        builder
            .buildNeuronCellModel<HebbState>("Hebb Network", hebbFactoryRecall)
            .buildLogger("recall_hebb_model_log.csv")
            .buildRootCoordinator()
            .startSimulation();
        
        builder.calculateRMSE<HebbState, NeuronHebbGridCell>(groundTruth, "hebb_model_log");
        break;
    }
    case SimulationType::Hopfield: {
        auto hopfieldFactoryRecall = std::make_shared<HopfieldNeuronFactory>(
            builder.greenImages,
            GeneralUtils::GaussianBlurFlatImagesWithPermutation(builder.greenImages, width, height, 5, 9, 0.5)
        );

        builder
            .buildNeuronCellModel<HopfieldState>("Hopfield Network", hopfieldFactoryRecall)
            .buildLogger("hopfield_model_log.csv")
            .buildRootCoordinator()
            .startSimulation();
        
        builder.calculateRMSE<HopfieldState, NeuronHopfieldGridCell>(groundTruth, "hopfield_model_log");
        break;
    }
    case SimulationType::ModernHopfield: {
        auto modernFactoryRecall = std::make_shared<ModernHopfieldNeuronFactory>(
            builder.greenImages,
            GeneralUtils::GaussianBlurFlatImagesWithPermutation(builder.greenImages, width, height, 5, 9, 0.5),
            true
        );

        builder
            .buildNeuronCellModel<ModernHopfieldState>("Modern Hopfield Network", modernFactoryRecall)
            .buildLogger("modern_hopfield_model_log.csv")
            .buildRootCoordinator()
            .startSimulation();
            
        builder.calculateRMSE<ModernHopfieldState, NeuronModernHopfieldGridCell>(groundTruth, "modern_hopfield_model_log");
        break;
    }
    case SimulationType::Field: {
        // Training 
        auto fieldFactoryTraining = std::make_shared<FieldNeuronFactory>(
            builder.greenImages,
            GeneralUtils::GaussianBlurFlatImagesWithPermutation(builder.greenImages, width, height, 5, 9, 0.5)
        );        

        builder
            .buildNeuronCellModel<FieldState>("Field Network", fieldFactoryTraining)
            .buildLogger("training_field_model_log.csv")
            .buildRootCoordinator()
            .startSimulation();


        // Get the training vector
        auto trainingVector = builder.extractTrainingParameters<FieldState, NeuronFieldGridCell>(4);

        // Recall
        auto fieldFactoryRecall = std::make_shared<FieldNeuronFactory>(
            builder.greenImages,
            GeneralUtils::GaussianBlurFlatImagesWithPermutation(builder.greenImages, width, height, 5, 9, 0.1),
            trainingVector
        );        

        builder
            .buildNeuronCellModel<FieldState>("Field Network", fieldFactoryRecall)
            .buildLogger("recall_field_model_log.csv")
            .buildRootCoordinator()
            .startSimulation();
            
        builder.calculateRMSE<FieldState, NeuronFieldGridCell>(groundTruth, "field_model_log");

        break;
    }
    default:
        throw std::runtime_error("Unknown simulation type selected.");
    }


    return 0;
}