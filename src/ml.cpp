#include "ml.hpp"

void trainMLModel() {
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    cv::Mat trainingData = (cv::Mat_<float>(4, 2) <<
        1.0, 2.0,
        2.0, 3.0,
        3.0, 1.0,
        5.0, 4.0);

    cv::Mat labels = (cv::Mat_<int>(4, 1) << 0, 1, 0, 1);

    knn->train(trainingData, cv::ml::ROW_SAMPLE, labels);

    std::cout << "Machine Learning Model Trained!" << std::endl;
}

void classifyFood(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image!" << std::endl;
        return;
    }

    // Example placeholder: Just showing the image for now
    cv::imshow("Food Classification", image);
    cv::waitKey(0);

    std::cout << "Classification complete!" << std::endl;
}
