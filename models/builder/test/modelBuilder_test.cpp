#include <gtest/gtest.h>
#include "../modelBuilder.hpp"
#include "../../cells/factory/gridCellFactoryBase.hpp"
#include "../../cells/types/neuronHebbGridCell.hpp"

// Dummy test images
std::vector<Eigen::VectorXd> images;
std::shared_ptr<std::unordered_map<std::vector<int>, double>> globalWeights;

// Dummy Hebb Factory for test
class DummyHebbFactory : public GridCellFactoryBase<HebbState> {
public:
    std::shared_ptr<GridCell<HebbState, double>> create(
        const coordinates& cellId,
        const std::shared_ptr<const GridCellConfig<HebbState, double>>& cellConfig
    ) const override {
        auto modifiableConfig = std::make_shared<GridCellConfig<HebbState, double>>(*cellConfig);
        HebbState state;

        int width = modifiableConfig->scenario->shape[0];
        int height = modifiableConfig->scenario->shape[1];
        int flatIndex = cellId[1] * width + cellId[0]; // (y * width + x)

        double pixelValue = images[0](flatIndex);
        if (cellId[0] % 2 == 0) {
            pixelValue *= -1;
        }

        state.activationStrength = pixelValue;
        state.coords = cellId;
        state.imageWidth = width;
        state.imageHeight = height;
        state.weights = globalWeights;

        modifiableConfig->state = state;

        return std::make_shared<NeuronHebbGridCell>(cellId, modifiableConfig, std::make_shared<HebbState>(state));
    }
};

class ModelBuilderHebbTestFixture : public ::testing::Test {
protected:
    std::shared_ptr<GridCellDEVSCoupled<HebbState, double>> neuronCellModel;

    void SetUp() override {
        InitModel();
    }

    void InitModel() {
        ModelBuilder modelBuilder("config/simulation_config.json");
        modelBuilder.loadImages();

        images.clear();
        for (int i = 0; i < modelBuilder.greenImages.cols(); ++i) {
            images.push_back(modelBuilder.greenImages.col(i));
        }

        // Initialize shared weights
        globalWeights = std::make_shared<std::unordered_map<std::vector<int>, double>>();
        int width = modelBuilder.getImageWidth();
        int height = modelBuilder.getImageHeight();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                (*globalWeights)[{x, y}] = 0.0;
            }
        }

        auto dummyFactory = std::make_shared<DummyHebbFactory>();
        modelBuilder.buildNeuronCellModel<HebbState>("HebbianTestModel", dummyFactory);

        neuronCellModel = std::get<std::shared_ptr<GridCellDEVSCoupled<HebbState, double>>>(modelBuilder.neuronCellModel);

        modelBuilder.buildRootCoordinator();
    }
};

TEST_F(ModelBuilderHebbTestFixture, TestModelInitialization) {
    ASSERT_NE(neuronCellModel, nullptr); // Model should be initialized
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
