#include "gui.hpp"

void showImage(const std::string& path) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image!" << std::endl;
        return;
    }

    cv::imshow("Display Image", image);
    cv::waitKey(0);
}
