/**
 * Machine Learning Model Training Implementation
 */

#include "model_trainer.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <filesystem>
#include <curl/curl.h>

namespace fs = std::filesystem;

namespace Training {

// Callback function for CURL download
size_t writeCallback(void* ptr, size_t size, size_t nmemb, std::ofstream* stream) {
    size_t written = 0;
    if (stream) {
        stream->write(static_cast<char*>(ptr), size * nmemb);
        written = size * nmemb;
    }
    return written;
}

ModelTrainer::ModelTrainer(std::shared_ptr<Data::WasteDatabase> database,
                           std::shared_ptr<Detection::FoodDetector> detector,
                           const std::string& trainingDataPath,
                           float learningRate)
    : m_database(database),
      m_detector(detector),
      m_trainingDataPath(trainingDataPath),
      m_isTraining(false),
      m_numTrainingSamples(0),
      m_numValidationSamples(0) {

    m_config.learningRate = learningRate;

    // Create directory structure for training data
    try {
        fs::path basePath(m_trainingDataPath);
        m_imagesPath = (basePath / "images").string();
        m_annotationsPath = (basePath / "annotations").string();
        m_checkpointsPath = (basePath / "checkpoints").string();

        // Create directories if they don't exist
        if (!fs::exists(basePath)) {
            fs::create_directories(basePath);
        }

        if (!fs::exists(m_imagesPath)) {
            fs::create_directories(m_imagesPath);
        }

        if (!fs::exists(m_annotationsPath)) {
            fs::create_directories(m_annotationsPath);
        }

        if (!fs::exists(m_checkpointsPath)) {
            fs::create_directories(m_checkpointsPath);
        }

        std::cout << "Created training directories at " << m_trainingDataPath << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating training directories: " << e.what() << std::endl;
    }
}

bool ModelTrainer::trainModel() {
    return trainModelWithConfig(m_config);
}

bool ModelTrainer::trainModelWithConfig(const TrainingConfig& config) {
    if (m_isTraining) {
        std::cerr << "Training already in progress" << std::endl;
        return false;
    }

    m_isTraining = true;
    std::cout << "Starting model training with configuration:" << std::endl
              << "- Batch size: " << config.batchSize << std::endl
              << "- Epochs: " << config.epochs << std::endl
              << "- Learning rate: " << config.learningRate << std::endl
              << "- Architecture: " << config.modelArchitecture << std::endl
              << "- Data augmentation: " << (config.useDataAugmentation ? "enabled" : "disabled") << std::endl
              << "- Validation split: " << config.validationSplit << std::endl;

    // Prepare training data
    int numSamples = prepareTrainingData();
    if (numSamples <= 0) {
        std::cerr << "Failed to prepare training data" << std::endl;
        m_isTraining = false;
        return false;
    }

    std::cout << "Prepared " << numSamples << " training samples" << std::endl;

    // In a real implementation, this would call into the appropriate
    // OpenCV DNN training functions or another ML framework.
    // For this example, we'll simulate the training process.

    // Simulate training process with loss decrease
    m_lastMetrics = TrainingMetrics();
    m_lastMetrics.trainingLoss.clear();
    m_lastMetrics.validationLoss.clear();

    float initialLoss = 5.0f;
    float finalLoss = 0.5f;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        // Simulate decreasing loss
        float progress = static_cast<float>(epoch) / config.epochs;
        float trainLoss = initialLoss - (initialLoss - finalLoss) * progress +
                       ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.2f;
        float validLoss = trainLoss * 1.2f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.2f;

        m_lastMetrics.trainingLoss.push_back(trainLoss);
        m_lastMetrics.validationLoss.push_back(validLoss);

        if (epoch % 10 == 0 || epoch == config.epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << config.epochs
                      << " - Loss: " << trainLoss
                      << " - Val Loss: " << validLoss << std::endl;
        }

        // Simulate training delay
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Simulate early stopping
        if (trainLoss < 0.6f && epoch > config.epochs / 2) {
            std::cout << "Early stopping triggered" << std::endl;
            break;
        }
    }

    // Simulate final metrics
    m_lastMetrics.finalPrecision = 0.85f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.1f;
    m_lastMetrics.finalRecall = 0.82f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.1f;
    m_lastMetrics.finalMeanAveragePrecision = 0.78f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.1f;

    std::cout << "Training completed" << std::endl
              << "- Final precision: " << m_lastMetrics.finalPrecision << std::endl
              << "- Final recall: " << m_lastMetrics.finalRecall << std::endl
              << "- Final mAP: " << m_lastMetrics.finalMeanAveragePrecision << std::endl;

    // Generate a simulated model file
    std::string modelPath = (fs::path(m_checkpointsPath) / "model_final.weights").string();
    std::ofstream modelFile(modelPath, std::ios::binary);
    if (modelFile.is_open()) {
        // Write some dummy bytes
        std::vector<char> dummyData(1024 * 1024, 0);  // 1MB dummy file
        modelFile.write(dummyData.data(), dummyData.size());
        modelFile.close();

        std::cout << "Saved model to " << modelPath << std::endl;

        // Update the detector with the new model
        if (m_detector) {
            m_detector->loadModel(modelPath);
        }
    }

    m_isTraining = false;
    return true;
}

int ModelTrainer::prepareTrainingData() {
    // This function would normally:
    // 1. Extract images from the database
    // 2. Create annotation files for supervised learning
    // 3. Split into training and validation sets
    // 4. Apply data augmentation if enabled

    // For this example, we'll simulate these steps

    std::cout << "Preparing training data..." << std::endl;

    // Clear previous training data
    m_trainingImagePaths.clear();
    m_validationImagePaths.clear();

    // Get entries from the database to use as training samples
    auto entries = m_database->getEntries();

    // Count entries with image files
    int validEntries = 0;
    for (const auto& entry : entries) {
        if (!entry.imageFilename.empty() && fs::exists(entry.imageFilename)) {
            validEntries++;
        }
    }

    if (validEntries == 0) {
        std::cout << "No valid entries with images found in database" << std::endl;

        // In a real system, we'd handle this better
        // For this example, create some simulated training data
        int simulatedSamples = 100;

        for (int i = 0; i < simulatedSamples; i++) {
            std::string imagePath = (fs::path(m_imagesPath) / ("simulated_" + std::to_string(i) + ".jpg")).string();

            // Create a simulated image file
            cv::Mat simulatedImage(416, 416, CV_8UC3, cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
            cv::rectangle(simulatedImage, cv::Rect(rand() % 200, rand() % 200, 100, 100), cv::Scalar(0, 255, 0), 2);
            cv::imwrite(imagePath, simulatedImage);

            // Create simulated annotations
            std::vector<Detection::FoodItem> annotations;
            Detection::FoodItem item;
            item.className = "simulated_food";
            item.boundingBox = cv::Rect(rand() % 200, rand() % 200, 100, 100);
            item.confidence = 1.0f;  // Ground truth
            annotations.push_back(item);

            saveAnnotations(imagePath, annotations);

            m_trainingImagePaths.push_back(imagePath);
        }

        m_numTrainingSamples = static_cast<int>(m_trainingImagePaths.size());
        m_numValidationSamples = 0;

        return m_numTrainingSamples;
    }

    // Process real entries
    for (const auto& entry : entries) {
        if (!entry.imageFilename.empty() && fs::exists(entry.imageFilename)) {
            // Load the image
            cv::Mat image = cv::imread(entry.imageFilename);
            if (image.empty()) {
                continue;
            }

            // Create a training sample
            std::string sampleName = "training_" + std::to_string(rand()) + ".jpg";
            std::string imagePath = (fs::path(m_imagesPath) / sampleName).string();

            // Copy image to training directory
            cv::imwrite(imagePath, image);

            // Create annotation
            std::vector<Detection::FoodItem> annotations;
            Detection::FoodItem item;
            item.className = entry.foodType;
            // Simulate a bounding box (in a real system, this would come from the actual detection)
            item.boundingBox = cv::Rect(10, 10, image.cols - 20, image.rows - 20);
            item.confidence = 1.0f;  // Ground truth
            annotations.push_back(item);

            saveAnnotations(imagePath, annotations);

            // Add to training or validation set
            if ((static_cast<float>(rand()) / RAND_MAX) < m_config.validationSplit) {
                m_validationImagePaths.push_back(imagePath);
            } else {
                m_trainingImagePaths.push_back(imagePath);
            }

            // Apply data augmentation if enabled
            if (m_config.useDataAugmentation) {
                std::vector<cv::Mat> augmentedImages = augmentImage(image);

                for (size_t i = 0; i < augmentedImages.size(); i++) {
                    std::string augName = "aug_" + std::to_string(i) + "_" + sampleName;
                    std::string augPath = (fs::path(m_imagesPath) / augName).string();

                    // Save augmented image
                    cv::imwrite(augPath, augmentedImages[i]);

                    // Create augmented annotation (same as original)
                    saveAnnotations(augPath, annotations);

                    // Add to training set
                    m_trainingImagePaths.push_back(augPath);
                }
            }
        }
    }

    // Shuffle training data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(m_trainingImagePaths.begin(), m_trainingImagePaths.end(), g);

    m_numTrainingSamples = static_cast<int>(m_trainingImagePaths.size());
    m_numValidationSamples = static_cast<int>(m_validationImagePaths.size());

    std::cout << "Prepared " << m_numTrainingSamples << " training samples and "
              << m_numValidationSamples << " validation samples" << std::endl;

    return m_numTrainingSamples + m_numValidationSamples;
}

std::vector<cv::Mat> ModelTrainer::augmentImage(const cv::Mat& image) {
    std::vector<cv::Mat> augmentedImages;

    if (image.empty()) {
        return augmentedImages;
    }

    // 1. Horizontal flip
    cv::Mat flipped;
    cv::flip(image, flipped, 1);
    augmentedImages.push_back(flipped);

    // 2. Rotation (slight)
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(
        cv::Point2f(image.cols / 2, image.rows / 2),
        10.0,  // 10 degrees
        1.0
    );
    cv::Mat rotated;
    cv::warpAffine(image, rotated, rotationMatrix, image.size());
    augmentedImages.push_back(rotated);

    // 3. Brightness adjustment
    cv::Mat brighter;
    image.convertTo(brighter, -1, 1.0, 30);  // Increase brightness
    augmentedImages.push_back(brighter);

    cv::Mat darker;
    image.convertTo(darker, -1, 1.0, -30);  // Decrease brightness
    augmentedImages.push_back(darker);

    // 4. Add noise
    cv::Mat noise = image.clone();
    cv::Mat randomNoise(image.size(), CV_8UC3);
    cv::randu(randomNoise, cv::Scalar(0, 0, 0), cv::Scalar(20, 20, 20));
    cv::add(image, randomNoise, noise);
    augmentedImages.push_back(noise);

    return augmentedImages;
}

bool ModelTrainer::saveAnnotations(const std::string& imagePath, const std::vector<Detection::FoodItem>& annotations) {
    // Extract base filename without extension
    fs::path path(imagePath);
    std::string baseName = path.stem().string();

    // Create annotation file path
    std::string annotationPath = (fs::path(m_annotationsPath) / (baseName + ".txt")).string();

    try {
        std::ofstream file(annotationPath);
        if (!file.is_open()) {
            std::cerr << "Failed to open annotation file: " << annotationPath << std::endl;
            return false;
        }

        // Get image dimensions (used for normalization)
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << imagePath << std::endl;
            return false;
        }

        int imageWidth = image.cols;
        int imageHeight = image.rows;

        // Write annotations in YOLO format:
        // <class_id> <center_x> <center_y> <width> <height>
        // Where coordinates are normalized to [0, 1]
        for (const auto& item : annotations) {
            // Get class ID from detector
            int classId = 0;
            auto classNames = m_detector->getClassNames();
            for (size_t i = 0; i < classNames.size(); i++) {
                if (classNames[i] == item.className) {
                    classId = static_cast<int>(i);
                    break;
                }
            }

            // Calculate normalized coordinates
            float centerX = (item.boundingBox.x + item.boundingBox.width / 2.0f) / imageWidth;
            float centerY = (item.boundingBox.y + item.boundingBox.height / 2.0f) / imageHeight;
            float width = static_cast<float>(item.boundingBox.width) / imageWidth;
            float height = static_cast<float>(item.boundingBox.height) / imageHeight;

            // Write to file
            file << classId << " "
                 << centerX << " "
                 << centerY << " "
                 << width << " "
                 << height << std::endl;
        }

        file.close();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving annotations: " << e.what() << std::endl;
        return false;
    }
}

TrainingMetrics ModelTrainer::getLastTrainingMetrics() const {
    return m_lastMetrics;
}

void ModelTrainer::setUseDataAugmentation(bool use) {
    m_config.useDataAugmentation = use;
}

bool ModelTrainer::getUseDataAugmentation() const {
    return m_config.useDataAugmentation;
}

bool ModelTrainer::downloadPretrainedModel(const std::string& modelUrl, const std::string& outputPath) {
    std::cout << "Downloading pretrained model from " << modelUrl << " to " << outputPath << std::endl;

    // Initialize CURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return false;
    }

    // Open output file
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputPath << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    // Configure CURL
    curl_easy_setopt(curl, CURLOPT_URL, modelUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3600L); // 1 hour timeout

    // Perform download
    CURLcode res = curl_easy_perform(curl);

    // Cleanup
    outFile.close();
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        // Remove partial file
        std::remove(outputPath.c_str());
        return false;
    }

    std::cout << "Download completed successfully" << std::endl;
    return true;
}

bool ModelTrainer::initializeFromPretrainedModel(const std::string& modelPath) {
    if (!fs::exists(modelPath)) {
        std::cerr << "Pretrained model file not found: " << modelPath << std::endl;
        return false;
    }

    std::cout << "Initializing from pretrained model: " << modelPath << std::endl;

    // In a real implementation, this would load the model
    // and prepare it for transfer learning

    // For this example, we'll just copy the file to our checkpoints directory
    std::string destPath = (fs::path(m_checkpointsPath) / "pretrained_base.weights").string();

    try {
        fs::copy_file(modelPath, destPath, fs::copy_options::overwrite_existing);

        // Update the detector
        if (m_detector) {
            return m_detector->loadModel(destPath);
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing from pretrained model: " << e.what() << std::endl;
        return false;
    }
}

float ModelTrainer::evaluateModel() {
    // In a real implementation, this would:
    // 1. Run the model on validation data
    // 2. Calculate precision, recall, mAP, etc.

    // For this example, we'll simulate evaluation

    if (m_numValidationSamples == 0) {
        std::cerr << "No validation samples available for evaluation" << std::endl;
        return 0.0f;
    }

    std::cout << "Evaluating model on " << m_numValidationSamples << " validation samples..." << std::endl;

    // Simulate evaluation metrics
    float precision = 0.85f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.1f;
    float recall = 0.82f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.1f;
    float mAP = 0.78f + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.1f;

    std::cout << "Evaluation results:" << std::endl
              << "- Precision: " << precision << std::endl
              << "- Recall: " << recall << std::endl
              << "- mAP: " << mAP << std::endl;

    return mAP;
}

void ModelTrainer::setLearningRate(float rate) {
    m_config.learningRate = rate;
}

float ModelTrainer::getLearningRate() const {
    return m_config.learningRate;
}

void ModelTrainer::setBatchSize(int size) {
    m_config.batchSize = size;
}

int ModelTrainer::getBatchSize() const {
    return m_config.batchSize;
}

void ModelTrainer::setEpochs(int epochs) {
    m_config.epochs = epochs;
}

int ModelTrainer::getEpochs() const {
    return m_config.epochs;
}

void ModelTrainer::onEpochEnd(void* userData, int epoch, float loss, float accuracy) {
    ModelTrainer* trainer = static_cast<ModelTrainer*>(userData);

    if (trainer) {
        // Update metrics
        trainer->m_lastMetrics.trainingLoss.push_back(loss);

        // Log progress
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << loss << " - Accuracy: " << accuracy << std::endl;
        }
    }
}