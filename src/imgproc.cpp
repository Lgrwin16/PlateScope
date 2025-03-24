#include "imgproc.hpp"

void applyFilters(const std::string& path) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image!" << std::endl;
        return;
    }

    cv::Mat gray, blurred, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);

    cv::imshow("Original", image);
    cv::imshow("Grayscale", gray);
    cv::imshow("Blurred", blurred);
    cv::imshow("Edges", edges);
    cv::waitKey(0);
}
