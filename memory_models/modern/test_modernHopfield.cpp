#include <gtest/gtest.h>
#include "ModernHopfield.hpp"
#include "../../utils/generalUtils.hpp"
#include <matplot/matplot.h>
#include "../../utils/imageLoader.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>


class ModernHopfieldAtomicFixture: public ::testing::Test
{
protected:
    std::shared_ptr<ModernHopfield> model;

    void SetUp() override 
    {
        InitModel();
    }

    void InitModel() 
    {
        std::vector<Eigen::VectorXd> memory_keys;
        std::vector<Eigen::VectorXd> memory_values;
        auto images = ImageLoader::LoadImages(100,100);

        for (auto& image : images) {
            auto vec = GeneralUtils::ParseImageToVectorXd(image);
            memory_keys.push_back(vec);
        }
        memory_values = memory_keys;
        model = std::make_shared<ModernHopfield>(memory_keys, memory_values, 2.0);
    }

};

TEST_F(ModernHopfieldAtomicFixture, TestModernHopfieldModelInit) {
    ASSERT_TRUE(1);
}


TEST_F(ModernHopfieldAtomicFixture, TestModernHopfieldModelRetrieve) {
    // Von size 4 @ 110
    // Image 100x100
    int self = 110;
    int left = 109;
    int right = 111;
    int top = 220;
    int bottom = 10;

    Eigen::VectorXi indices(5);  
    indices << self, left, right, top, bottom;
    
    Eigen::VectorXd states(5);
    states << 0.005, 0.5, 1.0, 0.0, 0.5;


    int N = model->getRowSize();
    for (int i =0; i < N ; i++) {
        std::cout << "image " << std::to_string(i) << std::endl;
        for (int j =0; j < 5 ; j++) {
            std::cout << model->getKey(i, indices(j)) << std::endl;
        }
    }
    

    auto pattern = model->retrieve(states, indices);

    std::cout << pattern << std::endl;

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv); 
    return RUN_ALL_TESTS();
}
