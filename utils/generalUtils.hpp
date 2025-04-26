


#ifndef GENERAL_UTILS_HPP
#define GENERAL_UTILS_HPP

#include <string>
#include <memory>
#include <sstream>
#include "../types/imageStructures.hpp"

class GeneralUtils {
public:
    static int signum(double num) {  
        return (num > 0) ? 1 : -1;
    }

    static std::string parseCellIndexToCadmiumId(int x, int y) {
        return "(" + std::to_string(x) + "," +  std::to_string(y) + ")";
    }

    static std::tuple<int, int> stringToIndices(const std::string& neighborStringID) {
       	// Remove the parentheses and split by the comma
		std::string stripped = neighborStringID.substr(1, neighborStringID.size() - 2);  // Remove "(" and ")"
		std::stringstream ss(stripped);
		int index1, index2;
		// Extract the two integers separated by the comma
		char comma;  // To discard the comma
		ss >> index1 >> comma >> index2;
	
		return std::make_tuple(index1, index2); 
    }

    static double calculateEnergyCostFunction(std::shared_ptr<WeightMatrix> globalWeightMatrix, std::shared_ptr<StateMatrix> globalStateMatrix) {
        double energy = 0;
        // Flatten state matrix
        auto flatenStateMatrix = globalStateMatrix->stateMatrix.reshaped();
        auto N = globalWeightMatrix->weightMatrix.rows();
        auto M = globalWeightMatrix->weightMatrix.cols();
        for (int i =0; i < N; i++){
			for (int j =0; j < M; j++){
				energy += globalWeightMatrix->getWeightAt(i, j) * flatenStateMatrix(i) * flatenStateMatrix(j);
			}
		}
        // Return normalized energy 
        return (-0.5 * energy) / (N*M);
    }

    static Eigen::VectorXd ParseImageToVectorXd(const std::vector<std::vector<unsigned char>>& imageRaw) {
        int row = imageRaw.size();
        int col = imageRaw[0].size();
    
        // Automatically initialize the vector with the correct size, filled with zeros
        auto vec = Eigen::VectorXd(row * col);
    
        // Flatten the image into a 1D vector (row-major order)
        int index = 0;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                // Normalize pixel value to range [0, 1] or convert to [-1, 1] for binary image
                double pixelValue = static_cast<double>(imageRaw[i][j]) / 255.0; // Normalize to [0, 1]
                
                // If you need a binary representation (-1, 1) instead of (0, 1), use:
                // double pixelValue = (imageRaw[i][j] > 127) ? 1.0 : -1.0;
    
                vec(index++) = pixelValue;  // Store in vector
            }
        }
        return vec;
    }
    
    // Helper function for clamping coordinates (edge handling)
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
    
        Eigen::MatrixXd blurredImages(numPixels, numImages); // Preallocate!
    
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
        // --- Input Validation ---
        if (imageHeight <= 0 || imageWidth <= 0) {
             throw std::invalid_argument("Image height and width must be positive.");
        }
        if (imageRaw.size() != static_cast<long long>(imageHeight * imageWidth)) { // Use long long for comparison robustness
            throw std::invalid_argument("Input vector size does not match imageHeight * imageWidth.");
        }
        // Kernel size and sigma checks remain the same
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw std::invalid_argument("Kernel size must be a positive odd integer.");
        }
        if (sigma <= 0.0) {
            throw std::invalid_argument("Sigma (standard deviation) must be positive.");
        }
    
        // --- 1. Generate Gaussian Kernel (This part is unchanged) ---
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
    
        // Normalize Kernel (This part is unchanged)
        if (sumKernel > 1e-9) { // Avoid division by zero or near-zero
            kernel /= sumKernel;
        } else {
            kernel.setZero();
            kernel(kernelRadius, kernelRadius) = 1.0; // Default to no blur
            // Or: throw std::runtime_error("Kernel sum too small, check sigma value.");
        }
    
        // --- 2. Apply Convolution ---
        // Output vector size is known
        Eigen::VectorXd blurredImageFlat = Eigen::VectorXd::Zero(imageRaw.size());
    
        for (int r = 0; r < imageHeight; ++r) { // Iterate through logical image rows
            for (int c = 0; c < imageWidth; ++c) { // Iterate through logical image columns
    
                double weightedSum = 0.0;
    
                // Apply kernel centered at logical pixel (r, c)
                for (int i = 0; i < kernelSize; ++i) { // Kernel row index (0 to kernelSize-1)
                    for (int j = 0; j < kernelSize; ++j) { // Kernel column index (0 to kernelSize-1)
    
                        // Corresponding image coordinates relative to top-left (0,0)
                        // i maps to row offset, j maps to column offset
                        int img_r = r + i - kernelRadius;
                        int img_c = c + j - kernelRadius;
    
                        // Edge Handling (Clamp to Edge / Replicate Border) - using imageHeight/Width
                        int clamped_r = clamp(img_r, 0, imageHeight - 1);
                        int clamped_c = clamp(img_c, 0, imageWidth - 1);
    
                        // *** Calculate FLAT index for the clamped coordinates ***
                        int inputFlatIndex = clamped_r * imageWidth + clamped_c;
    
                        // *** Get image pixel value from the FLAT input vector ***
                        // No static_cast needed as imageRaw is already VectorXd (doubles)
                        double pixelValue = imageRaw(inputFlatIndex);
    
                        // Accumulate weighted sum (kernel access uses 0-based indices i,j)
                        weightedSum += pixelValue * kernel(i, j);
                    }
                }
    
                // --- 3. Assign to Flattened Output ---
                // Calculate the FLAT index for the current output pixel (r, c)
                int outputFlatIndex = r * imageWidth + c;
                blurredImageFlat(outputFlatIndex) = weightedSum;
            }
        }
    
        return blurredImageFlat;
    }

};

#endif
