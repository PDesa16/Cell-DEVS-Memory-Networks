


#ifndef GENERAL_UTILS_HPP
#define GENERAL_UTILS_HPP

#include <string>
#include <memory>
#include <sstream>
#include "../types/imageStructures.hpp"
#include "stochastic/random.hpp"

class GeneralUtils {
public:
    static int signum(double num) {  
        return (num > 0) ? 1 : -1;
    }

    static std::string parseCellIndexToCadmiumId(int x, int y) {
        return "(" + std::to_string(x) + "," +  std::to_string(y) + ")";
    }

    static std::tuple<int, int> stringToIndices(const std::string& neighborStringID) {
		std::string stripped = neighborStringID.substr(1, neighborStringID.size() - 2);  // Remove "(" and ")"
		std::stringstream ss(stripped);
		int index1, index2;
		char comma; 
		ss >> index1 >> comma >> index2;
	
		return std::make_tuple(index1, index2); 
    }

    static double calculateEnergyCostFunction(std::shared_ptr<WeightMatrix> globalWeightMatrix, std::shared_ptr<StateMatrix> globalStateMatrix) {
        double energy = 0;
        auto flatenStateMatrix = globalStateMatrix->stateMatrix.reshaped();
        auto N = globalWeightMatrix->weightMatrix.rows();
        auto M = globalWeightMatrix->weightMatrix.cols();
        for (int i =0; i < N; i++){
			for (int j =0; j < M; j++){
				energy += globalWeightMatrix->getWeightAt(i, j) * flatenStateMatrix(i) * flatenStateMatrix(j);
			}
		}
        return (-0.5 * energy) / (N*M);
    }

    static Eigen::VectorXd ParseImageToVectorXd(const std::vector<std::vector<unsigned char>>& imageRaw) {
        int row = imageRaw.size();
        int col = imageRaw[0].size();
    
        auto vec = Eigen::VectorXd(row * col);
    
        int index = 0;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                double pixelValue = static_cast<double>(imageRaw[i][j]) / 255.0;
                vec(index++) = pixelValue; 
            }
        }
        return vec;
    }
    
    static inline int clamp(int val, int min_val, int max_val) {
        return std::max(min_val, std::min(val, max_val));
    }


    static Eigen::MatrixXd GaussianBlurFlatImages(
        const Eigen::MatrixXd& imagesRaw,
        int imageHeight,
        int imageWidth,
        double sigma,
        int kernelSize)
    {
        int numPixels = imageHeight * imageWidth;
        int numImages = imagesRaw.cols();
    
        Eigen::MatrixXd blurredImages(numPixels, numImages); 
    
        for (int i = 0; i < numImages; ++i) {
            blurredImages.col(i) = GeneralUtils::GaussianBlurFlatImage(
                imagesRaw.col(i), imageHeight, imageWidth, sigma, kernelSize
            );
        }
    
        return blurredImages;
    }
    

    static Eigen::VectorXd GaussianBlurFlatImage(
        const Eigen::VectorXd& imageRaw, 
        int imageHeight,                 
        int imageWidth,                 
        double sigma,
        int kernelSize)
    {

        if (imageHeight <= 0 || imageWidth <= 0) {
             throw std::invalid_argument("Image height and width must be positive.");
        }
        if (imageRaw.size() != static_cast<long long>(imageHeight * imageWidth)) { 
            throw std::invalid_argument("Input vector size does not match imageHeight * imageWidth.");
        }
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw std::invalid_argument("Kernel size must be a positive odd integer.");
        }
        if (sigma <= 0.0) {
            throw std::invalid_argument("Sigma (standard deviation) must be positive.");
        }
    
        int kernelRadius = kernelSize / 2;
        Eigen::MatrixXd kernel = Eigen::MatrixXd::Zero(kernelSize, kernelSize);
        double sumKernel = 0.0;
        double twoSigmaSq = 2.0 * sigma * sigma;
    
        for (int i = -kernelRadius; i <= kernelRadius; ++i) {
            for (int j = -kernelRadius; j <= kernelRadius; ++j) {
                double value = std::exp(-(static_cast<double>(i * i + j * j)) / twoSigmaSq);
                kernel(i + kernelRadius, j + kernelRadius) = value;
                sumKernel += value;
            }
        }
    
        if (sumKernel > 1e-9) { 
            kernel /= sumKernel;
        } else {
            kernel.setZero();
            kernel(kernelRadius, kernelRadius) = 1.0; 
        }
    

        Eigen::VectorXd blurredImageFlat = Eigen::VectorXd::Zero(imageRaw.size());
    
        for (int r = 0; r < imageHeight; ++r) {
            for (int c = 0; c < imageWidth; ++c) {
    
                double weightedSum = 0.0;
    

                for (int i = 0; i < kernelSize; ++i) { 
                    for (int j = 0; j < kernelSize; ++j) { 
    
                        int img_r = r + i - kernelRadius;
                        int img_c = c + j - kernelRadius;
    
                        int clamped_r = clamp(img_r, 0, imageHeight - 1);
                        int clamped_c = clamp(img_c, 0, imageWidth - 1);
    
                        int inputFlatIndex = clamped_r * imageWidth + clamped_c;
    
                        double pixelValue = imageRaw(inputFlatIndex);
    
                        weightedSum += pixelValue * kernel(i, j);
                    }
                }
    
                int outputFlatIndex = r * imageWidth + c;
                blurredImageFlat(outputFlatIndex) = weightedSum;
            }
        }
    
        return blurredImageFlat;
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static Eigen::MatrixXd calculateWeightMatrix(const Eigen::MatrixXd& patterns) {
        if (patterns.cols() == 0) return Eigen::MatrixXd(); 
    
        int featureDim = patterns.rows();
    
        Eigen::MatrixXd weightMatrix = Eigen::MatrixXd::Zero(featureDim, featureDim);
    
        double factor = 1.0 / static_cast<double>(patterns.cols()); 
    
        for (int i = 0; i < patterns.cols(); ++i) {
            Eigen::VectorXd vec = patterns.col(i).transpose(); 
            weightMatrix += factor * (vec * vec.transpose());  
        }
    
        weightMatrix.diagonal().setZero(); 
    
        return weightMatrix;
    }

    static Eigen::MatrixXd GaussianBlurFlatImagesWithPermutation(
        const Eigen::MatrixXd& input,
        int width, int height,
        int blurWidth, int blurHeight,
        double permutationRatio,
        unsigned int seed = 42
    ) {

        Eigen::MatrixXd blurred = GaussianBlurFlatImages(input, width, height, blurWidth, blurHeight);
    
        int numPixels = blurred.rows();
        int numImages = blurred.cols();

        RandomNumberGeneratorDEVS::seed(seed);
    
        int numPermuted = static_cast<int>(permutationRatio * numPixels);
    
        for (int n = 0; n < numImages; ++n) {
            for (int k = 0; k < numPermuted; ++k) {
                int idx1 = RandomNumberGeneratorDEVS::generateUniformInt(0, numPixels - 1);
                int idx2 = RandomNumberGeneratorDEVS::generateUniformInt(0, numPixels - 1);
    
                std::swap(blurred(idx1, n), blurred(idx2, n));
            }
        }
    
        return blurred;
    }
    
    

};

#endif
