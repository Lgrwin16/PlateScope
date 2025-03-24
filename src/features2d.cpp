#include "features2d.hpp"

void detectFeatures(const std::string& path) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image!" << std::endl;
        return;
    }

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);
    cv::imshow("Keypoints", output);
    cv::waitKey(0);
}
