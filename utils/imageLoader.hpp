
#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#define STB_IMAGE_IMPLEMENTATION
#include "../c_libs_image/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../c_libs_image/stb_image_resize2.h"
#include <iostream>
#include <filesystem>  
#include <vector>
#include <string>

namespace fs = std::filesystem; 

class ImageLoader {
public:
    static std::vector<std::vector<std::vector<unsigned char>>> LoadImages(int imageWidthDesired, int imageLengthDesired) {
        std::vector<std::vector<std::vector<unsigned char>>> imageSet;
        std::string directoryPath = "images";  

        if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
            std::cerr << "Error: Directory does not exist!" << std::endl;
            return imageSet;
        }

        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            int width, height, channels;

            if (entry.is_regular_file()) {
                std::string filePath = entry.path().string();
                unsigned char* image = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_grey);

                if (image == nullptr) {
                    std::cerr << "Error: Could not load " << filePath << std::endl;
                    continue;
                }
    
                unsigned char* resizedImage = resizeImage(image, width, height, imageWidthDesired , imageLengthDesired);
                
                if (!resizedImage) {
                    std::cerr << "Error: Could not resize " << filePath << std::endl;
                    stbi_image_free(image);
                    continue;
                }

                auto imageMatrix = ToMatrix(resizedImage, imageWidthDesired, imageLengthDesired);
                imageSet.push_back(imageMatrix);

                free(resizedImage); 
                stbi_image_free(image);
            }
        }

        return imageSet;
    }

    static std::vector<std::vector<std::vector<std::vector<unsigned char>>>> LoadImagesRGB(
        int imageWidthDesired,
        int imageLengthDesired
    ) {
        std::vector<std::vector<std::vector<std::vector<unsigned char>>>> imageSet;
        std::string directoryPath = "images";
    
        if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
            std::cerr << "Error: Directory does not exist!" << std::endl;
            return imageSet;
        }
    
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            int width, height, channels;
    
            if (!entry.is_regular_file())
                continue;
    
            std::string filePath = entry.path().string();
            unsigned char* image = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb);
    
            if (image == nullptr) {
                std::cerr << "Error: Could not load " << filePath << std::endl;
                continue;
            }
    
            unsigned char* resizedImage = resizeImageRGB(image, width, height, imageWidthDesired, imageLengthDesired);
            if (!resizedImage) {
                std::cerr << "Error: Could not resize " << filePath << std::endl;
                stbi_image_free(image);
                continue;
            }
    
            // 3 color channels, each with a matrix of imageLengthDesired x imageWidthDesired
            std::vector<std::vector<std::vector<unsigned char>>> separatedRGB(3, std::vector<std::vector<unsigned char>>(
                imageLengthDesired, std::vector<unsigned char>(imageWidthDesired)));
    
            for (int y = 0; y < imageLengthDesired; ++y) {
                for (int x = 0; x < imageWidthDesired; ++x) {
                    int idx = (y * imageWidthDesired + x) * 3;
                    separatedRGB[0][y][x] = resizedImage[idx + 0]; // Red
                    separatedRGB[1][y][x] = resizedImage[idx + 1]; // Green
                    separatedRGB[2][y][x] = resizedImage[idx + 2]; // Blue
                }
            }
    
            imageSet.push_back(separatedRGB);
    
            free(resizedImage);
            stbi_image_free(image);
        }
    
        return imageSet;
    }
     

    static unsigned char* resizeImage(unsigned char* image, int origWidth, int origHeight, int newWidth, int newHeight) {
        unsigned char* resizedImage = (unsigned char*)malloc(newWidth * newHeight);

        if (!resizedImage) {
            std::cerr << "Error: Memory allocation failed for resized image!" << std::endl;
            return nullptr;
        }

        if (!stbir_resize_uint8_srgb(image, origWidth, origHeight, origWidth, resizedImage, newWidth, newHeight, newWidth, STBIR_1CHANNEL)) {
            std::cerr << "Error: Image resizing failed!" << std::endl;
            free(resizedImage);
            return nullptr;
        }

        std::cout << "Resized image to: " << newWidth << "x" << newHeight << std::endl;
        return resizedImage;
    }

    static unsigned char* resizeImageRGB(unsigned char* image, int origWidth, int origHeight, int newWidth, int newHeight) {
        const int channels = 3;
        unsigned char* resizedImage = (unsigned char*)malloc(newWidth * newHeight * channels);
    
        if (!resizedImage) {
            std::cerr << "Error: Memory allocation failed for resized RGB image!" << std::endl;
            return nullptr;
        }
    
        if (!stbir_resize_uint8_srgb(
                image, origWidth, origHeight, origWidth * channels,
                resizedImage, newWidth, newHeight, newWidth * channels,
                STBIR_RGB
            )) {
            std::cerr << "Error: RGB image resizing failed!" << std::endl;
            free(resizedImage);
            return nullptr;
        }
    
        std::cout << "Resized RGB image to: " << newWidth << "x" << newHeight << std::endl;
        return resizedImage;
    }
    

    static std::vector<std::vector<unsigned char>> ToMatrix(unsigned char* imageRaw, int width, int height) {
        std::vector<std::vector<unsigned char>> imageMatrix(height, std::vector<unsigned char>(width));

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                imageMatrix[i][j] = imageRaw[i * width + j];
            }
        }
        return imageMatrix;
    }

};

#endif
