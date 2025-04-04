/**
 * Food Detection Module Header
 *
 * Uses machine learning models to detect and classify food items in images
 */

#ifndef FOOD_DETECTOR_H
#define FOOD_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace Detection {

// Structure to represent a detected food item
struct FoodItem {
    std::string className;       // Food class name
    float confidence;            // Detection confidence
    cv::Rect boundingBox;        // Location in the image
    float estimatedWeight;       // Estimated weight in grams
    bool isWaste;                // Flag for waste vs. non-waste
    std::string timestamp;       // Detection timestamp

    FoodItem() : confidence(0.0f), estimatedWeight(0.0f), isWaste(false) {}
};

// A collection of detected items in a single frame
using DetectionResult = std::vector<FoodItem>;

class FoodDetector {
public:
    FoodDetector(const std::string& modelPath,
                 const std::string& classesPath,
                 float confidenceThreshold = 0.5f);

    // Core detection functions
    DetectionResult detectFoodWaste(const cv::Mat& frame);

    // Model management
    bool loadModel(const std::string& modelPath);
    bool saveModel(const std::string& modelPath);
    void updateModel(const cv::dnn::Net& newModel);

    // Detection settings
    void setConfidenceThreshold(float threshold);
    float getConfidenceThreshold() const;

    // Class management
    std::vector<std::string> getClassNames() const;
    int getNumClasses() const;
    bool addClass(const std::string& className);

    // Food waste estimation
    float estimateWeight(const cv::Rect& bbox, const std::string& foodClass) const;

private:
    // Pre-processing for detection
    cv::Mat preProcessFrame(const cv::Mat& frame);

    // Post-processing of network outputs
    DetectionResult processDetections(const std::vector<cv::Mat>& outputs, const cv::Mat& frame);

    // Classification functions
    bool isWasteItem(const cv::Mat& foodROI, const std::string& foodClass) const;

    // Load class names from file
    bool loadClasses(const std::string& classesPath);

    // Deep neural network
    cv::dnn::Net m_net;

    // Detection parameters
    float m_confidenceThreshold;
    float m_nmsThreshold;  // Non-maximum suppression threshold

    // Input configuration
    cv::Size m_inputSize;
    float m_scale;
    cv::Scalar m_mean;

    // Class names
    std::vector<std::string> m_classNames;

    // Food reference data (for weight estimation)
    std::map<std::string, float> m_referenceWeights;

    // Output layer names
    std::vector<std::string> m_outputLayerNames;
};

} // namespace Detection

#endif // FOOD_DETECTOR_H