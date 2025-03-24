#include "objdetect.hpp"

void detectFaces(const std::string& path, const std::string& cascadePath) {
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        std::cerr << "Error: Could not load face cascade!" << std::endl;
        return;
    }

    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image!" << std::endl;
        return;
    }

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 4, 0, cv::Size(30, 30));

    for (const auto& face : faces) {
        cv::rectangle(image, face, cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Detected Faces", image);
    cv::waitKey(0);
}
