#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#define IMAGE_H 224
#define IMAGE_W 224

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* processImage: Processes image passed to be used as output. Returns processed image
*/
cv::Mat processImage(cv::Mat image)
{
    cv::Mat processImage;

    // Resize
    cv::resize(image, processImage, cv::Size(IMAGE_W, IMAGE_H), cv::INTER_LINEAR);

    // Channels
    cv::cvtColor(processImage, processImage, cv::COLOR_BGR2RGB);

    return processImage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* convertToTensor: Converts an image to tensor
*/
torch::Tensor convertToTensor(cv::Mat image)
{
    // Convert to tensor
    torch::Tensor inputTensor = torch::from_blob(image.data, { 1, IMAGE_H, IMAGE_W, 3 }, torch::kByte);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat);
    inputTensor = inputTensor.div(255); // Normalize the pixel values

    return inputTensor;
}

int main() {
    // Load model
    torch::jit::script::Module model = torch::jit::load("F:/myzesty/output_model.pt");

    // Load images
    std::vector<cv::String> imageFiles;
    cv::glob("F:/myzesty/*.jpg", imageFiles, false);

    for (const auto& imageFile : imageFiles) {
        cv::Mat image = cv::imread(imageFile);
        cv::Mat processedImage = processImage(image);

        if (image.empty()) {
            std::cerr << "Error: Unable to read the image file " << imageFile << std::endl;
            continue;
        }

        // Execute inference
        torch::NoGradGuard no_grad;
        auto outputTuple = model.forward({ convertToTensor(processedImage) }).toTuple();
        at::Tensor output = outputTuple->elements()[0].toTensor();

        // Post-process the output tensor
        // Example: thresholding
        auto threshold = torch::tensor(0.5); // Adjust threshold value as needed
        at::Tensor binaryMask = (output > threshold).to(torch::kByte);

        // Convert the tensor to a NumPy array for displaying
        at::TensorOptions options = output.options();
        binaryMask = binaryMask.squeeze().to(torch::kCPU).to(torch::kF32).mul(255).to(torch::kU8);
        cv::Mat binaryMat(cv::Size(IMAGE_W, IMAGE_H), CV_8UC1, binaryMask.data_ptr());

        // Apply binary mask to original image
        cv::Mat result;
        cv::bitwise_and(processedImage, processedImage, result, binaryMat);

        // Display the result
        cv::imshow("Input Image", image);
        cv::imshow("Binary Mask", binaryMat);
        cv::imshow("Result", result);
        cv::waitKey(0);
    }

    return 0;
}
