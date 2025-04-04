/**
 * FoodWasteMonitor - Main Application
 *
 * This program uses computer vision and machine learning to detect and
 * track food waste in college dining halls, providing statistical analysis
 * on waste patterns.
 */

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "camera/camera_manager.h"
#include "detection/food_detector.h"
#include "data/waste_database.h"
#include "analysis/stats_analyzer.h"
#include "training/model_trainer.h"
#include "ui/user_interface.h"
#include "utils/config_loader.h"

int main(int argc, char* argv[]) {
    std::cout << "Starting Food Waste Monitoring System..." << std::endl;

    try {
        // Load configuration
        Utils::ConfigLoader config("config.json");

        // Initialize components
        auto cameraManager = std::make_shared<Camera::CameraManager>(config.getCameraIndex());
        auto database = std::make_shared<Data::WasteDatabase>(config.getDatabasePath());
        auto detector = std::make_shared<Detection::FoodDetector>(
            config.getModelPath(),
            config.getClassesPath(),
            config.getConfidenceThreshold()
        );
        auto analyzer = std::make_shared<Analysis::StatsAnalyzer>(database);
        auto trainer = std::make_shared<Training::ModelTrainer>(
            database,
            detector,
            config.getTrainingDataPath(),
            config.getLearningRate()
        );
        auto ui = std::make_shared<UI::UserInterface>(
            cameraManager,
            detector,
            analyzer,
            trainer,
            config
        );

        // Main processing loop
        ui->start();

        // Scheduled periodic training
        auto lastTrainingTime = std::chrono::steady_clock::now();

        while (ui->isRunning()) {
            // Process current frame
            if (cameraManager->hasNewFrame()) {
                cv::Mat frame = cameraManager->getLatestFrame();

                // Detect food waste in the frame
                auto detectionResults = detector->detectFoodWaste(frame);

                // Update database with new detections
                if (!detectionResults.empty()) {
                    database->addDetections(detectionResults);
                    analyzer->updateStats();
                }

                // Display processed frame with detections
                ui->updateFrame(frame, detectionResults);
            }

            // Check if it's time for periodic training
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsedHours = std::chrono::duration_cast<std::chrono::hours>(
                currentTime - lastTrainingTime
            ).count();

            if (elapsedHours >= config.getTrainingIntervalHours()) {
                std::cout << "Starting periodic model training..." << std::endl;
                trainer->trainModel();
                lastTrainingTime = currentTime;
                std::cout << "Model training completed." << std::endl;
            }

            // Handle user input
            ui->processEvents();
        }

        // Save final data before exit
        database->saveToFile();
        detector->saveModel(config.getModelPath());

        std::cout << "Food Waste Monitoring System shut down successfully." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}