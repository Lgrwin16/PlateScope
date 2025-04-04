/**
 * Food Detection Implementation
 */

#include "food_detector.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace Detection {

FoodDetector::FoodDetector(const std::string& modelPath,
                           const std::string& classesPath,
                           float confidenceThreshold)
    : m_confidenceThreshold(confidenceThreshold),
      m_nmsThreshold(0.4f),
      m_inputSize(416, 416),
      m_scale(1/255.0f),
      m_mean(0, 0, 0) {

    // Load the model and classes
    if (!loadModel(modelPath)) {
        throw std::runtime_error("Failed to load detection model from: " + modelPath);
    }

    if (!loadClasses(classesPath)) {
        throw std::runtime_error("Failed to load class names from: " + classesPath);
    }

    // Initialize reference weights for common food items (in grams)
    // These are approximate average weights used for estimation
    m_referenceWeights = {
        {"apple", 150.0f},
        {"banana", 120.0f},
        {"bread", 40.0f},
        {"burger", 150.0f},
        {"cake", 100.0f},
        {"carrot", 60.0f},
        {"chicken", 200.0f},
        {"cookie", 30.0f},
        {"fries", 100.0f},
        {"pizza", 100.0f},
        {"rice", 150.0f},
        {"salad", 200.0f},
        {"sandwich", 180.0f},
        {"pasta", 180.0f},
        {"vegetable", 80.0f}
    };

    std::cout << "Food detector initialized with " << m_classNames.size() << " classes" << std::endl;
}

DetectionResult FoodDetector::detectFoodWaste(const cv::Mat& frame) {
    if (frame.empty()) {
        return DetectionResult();
    }

    // Pre-process the frame
    cv::Mat blob = preProcessFrame(frame);

    // Set the input to the network
    m_net.setInput(blob);

    // Forward pass - run the network
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_outputLayerNames);

    // Process the network outputs
    DetectionResult result = processDetections(outputs, frame);

    // Add timestamp to each detection
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S");
    std::string timestamp = ss.str();

    for (auto& item : result) {
        item.timestamp = timestamp;
    }

    return result;
}

cv::Mat FoodDetector::preProcessFrame(const cv::Mat& frame) {
    // Create a blob from the image
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        m_scale,
        m_inputSize,
        m_mean,
        true,  // swapRB
        false  // crop
    );

    return blob;
}

DetectionResult FoodDetector::processDetections(const std::vector<cv::Mat>& outputs, const cv::Mat& frame) {
    DetectionResult results;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Process each detection output
    for (const auto& output : outputs) {
        // Each row is a detection with classes scores
        for (int i = 0; i < output.rows; i++) {
            // Find the class with maximum score
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

            if (confidence > m_confidenceThreshold) {
                // Get the bounding box
                int centerX = static_cast<int>(output.at<float>(i, 0) * frame.cols);
                int centerY = static_cast<int>(output.at<float>(i, 1) * frame.rows);
                int width = static_cast<int>(output.at<float>(i, 2) * frame.cols);
                int height = static_cast<int>(output.at<float>(i, 3) * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Apply non-maximum suppression to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confidenceThreshold, m_nmsThreshold, indices);

    // Process the final detections
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        // Make sure the box is within the frame boundaries
        box.x = std::max(0, box.x);
        box.y = std::max(0, box.y);
        box.width = std::min(box.width, frame.cols - box.x);
        box.height = std::min(box.height, frame.rows - box.y);

        // Create a food item from the detection
        if (box.width > 0 && box.height > 0 && classIds[idx] < static_cast<int>(m_classNames.size())) {
            FoodItem item;
            item.className = m_classNames[classIds[idx]];
            item.confidence = confidences[idx];
            item.boundingBox = box;

            // Extract the ROI for waste classification
            cv::Mat roi = frame(box);
            item.isWaste = isWasteItem(roi, item.className);

            // Only include waste items in the results
            if (item.isWaste) {
                // Estimate the weight based on the size of the bounding box
                item.estimatedWeight = estimateWeight(box, item.className);
                results.push_back(item);
            }
        }
    }

    return results;
}

bool FoodDetector::isWasteItem(const cv::Mat& foodROI, const std::string& foodClass) const {
    // This is a simplified implementation
    // In a real-world scenario, you would use another classifier here
    // to determine if the food item is waste or not

    // For simulation purposes, let's use simple image analysis
    // For example, assume that items with low saturation/value might be waste
    cv::Mat hsvROI;
    cv::cvtColor(foodROI, hsvROI, cv::COLOR_BGR2HSV);

    // Calculate average saturation and value
    cv::Scalar mean = cv::mean(hsvROI);
    float avgSaturation = mean[1];
    float avgValue = mean[2];

    // Simple threshold-based decision
    // In reality, this would be a more sophisticated classifier
    if (avgSaturation < 50 || avgValue < 100) {
        return true; // Low saturation or darkness might indicate waste
    }

    // Random factor for simulation - in real implementation this would be removed
    // Makes approximately 30% of food detected as waste
    return (rand() % 100) < 30;
}

float FoodDetector::estimateWeight(const cv::Rect& bbox, const std::string& foodClass) const {
    // Get the reference weight for the food class or use a default
    float referenceWeight = 100.0f; // Default weight in grams
    auto it = m_referenceWeights.find(foodClass);
    if (it != m_referenceWeights.end()) {
        referenceWeight = it->second;
    }

    // Calculate a size factor based on the bounding box area
    // This is a simple approximation - in a real system you would use
    // depth information or calibrated measurements
    float area = static_cast<float>(bbox.width * bbox.height);
    float referenceArea = 10000.0f; // Reference area in pixels
    float sizeFactor = area / referenceArea;

    // Estimate the weight
    float estimatedWeight = referenceWeight * sizeFactor;

    // Add some reasonable limits
    estimatedWeight = std::max(5.0f, std::min(estimatedWeight, 1000.0f));

    return estimatedWeight;
}

bool FoodDetector::loadModel(const std::string& modelPath) {
    try {
        // Load the network
        m_net = cv::dnn::readNet(modelPath);

        // Check if using CUDA is possible
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using CUDA for inference" << std::endl;
        } else {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "Using CPU for inference" << std::endl;
        }

        // Get output layer names
        m_outputLayerNames = m_net.getUnconnectedOutLayersNames();

        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool FoodDetector::saveModel(const std::string& modelPath) {
    try {
        // For models that support serialization
        m_net.save(modelPath);
        std::cout << "Model saved to " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

void FoodDetector::updateModel(const cv::dnn::Net& newModel) {
    m_net = newModel;
    m_outputLayerNames = m_net.getUnconnectedOutLayersNames();
}

bool FoodDetector::loadClasses(const std::string& classesPath) {
    std::ifstream file(classesPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open classes file: " << classesPath << std::endl;
        return false;
    }

    m_classNames.clear();
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (!line.empty()) {
            m_classNames.push_back(line);
        }
    }

    return !m_classNames.empty();
}

void FoodDetector::setConfidenceThreshold(float threshold) {
    m_confidenceThreshold = threshold;
}

float FoodDetector::getConfidenceThreshold() const {
    return m_confidenceThreshold;
}

std::vector<std::string> FoodDetector::getClassNames() const {
    return m_classNames;
}

int FoodDetector::getNumClasses() const {
    return static_cast<int>(m_classNames.size());
}

bool FoodDetector::addClass(const std::string& className) {
    // Check if class already exists
    for (const auto& name : m_classNames) {
        if (name == className) {
            return false;
        }
    }

    // Add the new class
    m_classNames.push_back(className);
    return true;
}

} // namespace Detection