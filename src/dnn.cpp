#include "dnn.hpp"

void runDNN(const std::string& model, const std::string& imagePath) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image!" << std::endl;
        return;
    }

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224, 224), cv::Scalar(), true, false);
    net.setInput(blob);
    cv::Mat output = net.forward();

    std::cout << "Model output: " << output << std::endl;
}
