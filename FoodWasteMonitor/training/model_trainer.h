/**
 * Machine Learning Model Training Module Header
 *
 * Manages training and updating of the food detection model
 */

#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "../detection/food_detector.h"
#include "../data/waste_database.h"

namespace Training {

// Training configuration structure
struct TrainingConfig {
    int batchSize;
    int epochs;
    float learningRate;
    std::string modelArchitecture;
    bool useDataAugmentation;
    float validationSplit;

    TrainingConfig()
        : batchSize(16),
          epochs(100),
          learningRate(0.001f),
          modelArchitecture("YOLOv4-tiny"),
          useDataAugmentation(true),
          validationSplit(0.2f) {
    }
};

// Training metrics structure
struct TrainingMetrics {
    std::vector<float> trainingLoss;
    std::vector<float> validationLoss;
    float finalPrecision;
    float finalRecall;
    float finalMeanAveragePrecision;

    TrainingMetrics()
        : finalPrecision(0.0f),
          finalRecall(0.0f),
          finalMeanAveragePrecision(0.0f) {
    }
};

class ModelTrainer {
public:
    ModelTrainer(std::shared_ptr<Data::WasteDatabase> database,
                 std::shared_ptr<Detection::FoodDetector> detector,
                 const std::string& trainingDataPath,
                 float learningRate = 0.001f);

    // Training functions
    bool trainModel();
    bool trainModelWithConfig(const TrainingConfig& config);

    // Data preparation
    int prepareTrainingData();

    // Training results
    TrainingMetrics getLastTrainingMetrics() const;

    // Data augmentation
    void setUseDataAugmentation(bool use);
    bool getUseDataAugmentation() const;

    // Transfer learning
    bool downloadPretrainedModel(const std::string& modelUrl, const std::string& outputPath);
    bool initializeFromPretrainedModel(const std::string& modelPath);

    // Model evaluation
    float evaluateModel();

    // Configuration
    void setLearningRate(float rate);
    float getLearningRate() const;
    void setBatchSize(int size);
    int getBatchSize() const;
    void setEpochs(int epochs);
    int getEpochs() const;

private:
    // Helper functions
    std::vector<cv::Mat> augmentImage(const cv::Mat& image);
    void prepareImageForTraining(const cv::Mat& image, const std::vector<Detection::FoodItem>& annotations);
    bool saveAnnotations(const std::string& imagePath, const std::vector<Detection::FoodItem>& annotations);

    // Callbacks for integrating with OpenCV DNN module
    static void onEpochEnd(void* userData, int epoch, float loss, float accuracy);

    // Pointers to other components
    std::shared_ptr<Data::WasteDatabase> m_database;
    std::shared_ptr<Detection::FoodDetector> m_detector;

    // Training configuration
    TrainingConfig m_config;

    // Training paths
    std::string m_trainingDataPath;
    std::string m_imagesPath;
    std::string m_annotationsPath;
    std::string m_checkpointsPath;

    // Training state
    bool m_isTraining;
    TrainingMetrics m_lastMetrics;
    int m_numTrainingSamples;
    int m_numValidationSamples;

    // Training data
    std::vector<std::string> m_trainingImagePaths;
    std::vector<std::string> m_validationImagePaths;
};

} // namespace Training

#endif // MODEL_TRAINER_H