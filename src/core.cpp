#include "core.hpp"

void displayMatrix() {
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    std::cout << "Matrix:\n" << matrix << std::endl;
}

void displayPoint() {
    cv::Point pt(10, 20);
    std::cout << "Point: (" << pt.x << ", " << pt.y << ")" << std::endl;
}

void displaySize() {
    cv::Size sz(100, 50);
    std::cout << "Size: " << sz.width << "x" << sz.height << std::endl;
}
