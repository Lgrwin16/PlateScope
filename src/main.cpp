#include <iostream>
#include "core.hpp"
#include "gui.hpp"
#include "imgproc.hpp"
#include "videoio.hpp"
#include "dnn.hpp"
#include "features2d.hpp"
#include "objdetect.hpp"
#include "ml.hpp"

int main() {
    std::string imagePath = "test.jpg";
    std::string faceCascadePath = "haarcascade_frontalface_default.xml";
    std::string modelPath = "models/food_model.onnx";

    std::cout << "=== OpenCV Project ===" << std::endl;

    // Core functions
    displayMatrix();
    displayPoint();
    displaySize();

    // GUI: Show Image
    showImage(imagePath);

    // Image Processing
    applyFilters(imagePath);

    // Video Capture
    captureVideo();

    // Object Detection (Face Detection)
    detectFaces(imagePath, faceCascadePath);

    // Feature Detection
    detectFeatures(imagePath);

    // Machine Learning - Object Detection
    runDNN(modelPath, imagePath);

    // Machine Learning - Train a Model
    trainMLModel();

    return 0;
}
