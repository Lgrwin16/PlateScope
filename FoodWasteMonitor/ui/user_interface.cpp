/**
 * User Interface Implementation
 */

#include "user_interface.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

namespace UI {

//
// DetectionVisualizer Implementation
//

DetectionVisualizer::DetectionVisualizer(std::shared_ptr<Detection::FoodDetector> detector)
    : m_detector(detector),
      m_showLabels(true),
      m_showConfidence(true),
      m_showWeight(true) {
}

void DetectionVisualizer::setDetections(const Detection::DetectionResult& detections) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_detections = detections;
}

void DetectionVisualizer::setShowLabels(bool show) {
    m_showLabels = show;
}

void DetectionVisualizer::setShowConfidence(bool show) {
    m_showConfidence = show;
}

void DetectionVisualizer::setShowWeight(bool show) {
    m_showWeight = show;
}

void DetectionVisualizer::render(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);

    for (const auto& item : m_detections) {
        // Draw bounding box
        cv::Scalar color;
        if (item.isWaste) {
            // Red for waste items
            color = cv::Scalar(0, 0, 255);
        } else {
            // Green for non-waste items
            color = cv::Scalar(0, 255, 0);
        }

        cv::rectangle(frame, item.boundingBox, color, 2);

        // Draw label if enabled
        if (m_showLabels) {
            std::stringstream ss;
            ss << item.className;

            if (m_showConfidence) {
                ss << " (" << std::fixed << std::setprecision(0) << (item.confidence * 100) << "%)";
            }

            if (m_showWeight && item.isWaste) {
                ss << " - " << std::fixed << std::setprecision(0) << item.estimatedWeight << "g";
            }

            std::string label = ss.str();

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            cv::rectangle(
                frame,
                cv::Point(item.boundingBox.x, item.boundingBox.y - textSize.height - 5),
                cv::Point(item.boundingBox.x + textSize.width, item.boundingBox.y),
                color,
                cv::FILLED
            );

            cv::putText(
                frame,
                label,
                cv::Point(item.boundingBox.x, item.boundingBox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1
            );
        }
    }
}

void DetectionVisualizer::update() {
    // Nothing to update here
}

void DetectionVisualizer::handleMouseEvent(int event, int x, int y) {
    // Detect if mouse is over a detection box for potential interaction
}

//
// StatsVisualizer Implementation
//

StatsVisualizer::StatsVisualizer(std::shared_ptr<Analysis::StatsAnalyzer> analyzer)
    : m_analyzer(analyzer),
      m_showTopWastedFoods(true),
      m_showWasteTrend(true),
      m_showWasteByMeal(true),
      m_showInsights(true) {

    update();
}

void StatsVisualizer::setShowTopWastedFoods(bool show) {
    m_showTopWastedFoods = show;
}

void StatsVisualizer::setShowWasteTrend(bool show) {
    m_showWasteTrend = show;
}

void StatsVisualizer::setShowWasteByMeal(bool show) {
    m_showWasteByMeal = show;
}

void StatsVisualizer::setShowInsights(bool show) {
    m_showInsights = show;
}

void StatsVisualizer::render(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Determine layout based on frame size
    int padding = 10;
    int y = padding;

    // Render title
    cv::putText(
        frame,
        "Food Waste Statistics",
        cv::Point(padding, y + 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255, 255, 255),
        2
    );

    y += 40;

    // Render insights at the top
    if (m_showInsights && !m_insights.empty()) {
        renderInsights(frame, padding, y);
        y += static_cast<int>(m_insights.size() * 25) + padding;
    }

    // Render top wasted foods
    if (m_showTopWastedFoods) {
        renderTopWastedFoods(frame, padding, y);
        y += 150 + padding;
    }

    // Render waste trend and waste by meal side by side if possible
    if (frame.cols >= 800) {
        int halfWidth = (frame.cols - 3 * padding) / 2;

        if (m_showWasteTrend) {
            renderWasteTrend(frame, padding, y);
        }

        if (m_showWasteByMeal) {
            renderWasteByMeal(frame, padding * 2 + halfWidth, y);
        }
    } else {
        // Otherwise render them vertically
        if (m_showWasteTrend) {
            renderWasteTrend(frame, padding, y);
            y += 150 + padding;
        }

        if (m_showWasteByMeal) {
            renderWasteByMeal(frame, padding, y);
            y += 150 + padding;
        }
    }
}

void StatsVisualizer::update() {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Get updated insights
    m_insights = m_analyzer->getInsights();
}

void StatsVisualizer::handleMouseEvent(int event, int x, int y) {
    // Handle mouse interactions for stats visualizations
}

void StatsVisualizer::renderTopWastedFoods(cv::Mat& frame, int x, int y) {
    // Get top wasted foods
    auto topFoods = m_analyzer->getTopWastedFoods(5);
    if (topFoods.empty()) {
        cv::putText(
            frame,
            "No waste data available",
            cv::Point(x, y + 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(200, 200, 200),
            1
        );
        return;
    }

    // Render title
    cv::putText(
        frame,
        "Top Wasted Foods",
        cv::Point(x, y + 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(200, 200, 200),
        1
    );

    y += 40;

    // Get waste by type for amounts
    auto wasteByType = m_analyzer->getWasteByType();

    // Calculate maximum weight for scaling
    float maxWeight = 0.0f;
    for (const auto& food : topFoods) {
        if (wasteByType.count(food) && wasteByType[food] > maxWeight) {
            maxWeight = wasteByType[food];
        }
    }

    // Avoid division by zero
    if (maxWeight == 0.0f) {
        maxWeight = 1.0f;
    }

    // Draw bars
    int barWidth = 40;
    int barHeight = 100;
    int spacing = 20;

    for (size_t i = 0; i < topFoods.size(); i++) {
        const auto& food = topFoods[i];
        float weight = wasteByType.count(food) ? wasteByType[food] : 0.0f;

        // Calculate bar height based on weight
        int height = static_cast<int>((weight / maxWeight) * barHeight);

        // Draw bar
        cv::rectangle(
            frame,
            cv::Point(x + i * (barWidth + spacing), y + (barHeight - height)),
            cv::Point(x + i * (barWidth + spacing) + barWidth, y + barHeight),
            cv::Scalar(0, 0, 255),
            cv::FILLED
        );

        // Draw food name
        cv::putText(
            frame,
            food,
            cv::Point(x + i * (barWidth + spacing), y + barHeight + 15),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(200, 200, 200),
            1
        );

        // Draw weight
        std::stringstream ss;
        ss << std::fixed << std::setprecision(0) << weight << "g";

        cv::putText(
            frame,
            ss.str(),
            cv::Point(x + i * (barWidth + spacing), y + (barHeight - height) - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(200, 200, 200),
            1
        );
    }
}

void StatsVisualizer::renderWasteTrend(cv::Mat& frame, int x, int y) {
    // Get trend data
    auto trend = m_analyzer->analyzeDailyTrend(7); // Last 7 days

    // Render title
    cv::putText(
        frame,
        "Waste Trend (Last 7 Days)",
        cv::Point(x, y + 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(200, 200, 200),
        1
    );

    y += 40;

    // Graph dimensions
    int graphWidth = 300;
    int graphHeight = 100;

    // Draw graph background
    cv::rectangle(
        frame,
        cv::Point(x, y),
        cv::Point(x + graphWidth, y + graphHeight),
        cv::Scalar(50, 50, 50),
        cv::FILLED
    );

    // Draw grid lines
    for (int i = 0; i < 4; i++) {
        int gridY = y + i * (graphHeight / 3);
        cv::line(
            frame,
            cv::Point(x, gridY),
            cv::Point(x + graphWidth, gridY),
            cv::Scalar(100, 100, 100),
            1
        );
    }

    // Plot trend
    if (!trend.values.empty()) {
        // Find maximum value for scaling
        float maxVal = *std::max_element(trend.values.begin(), trend.values.end());
        if (maxVal < 1.0f) maxVal = 1.0f; // Avoid division by zero

        // Plot points and lines
        std::vector<cv::Point> points;
        int numPoints = static_cast<int>(trend.values.size());
        int pointSpacing = graphWidth / std::max(1, numPoints - 1);

        for (int i = 0; i < numPoints; i++) {
            int pointX = x + i * pointSpacing;
            int pointY = y + graphHeight - static_cast<int>((trend.values[i] / maxVal) * graphHeight);
            points.push_back(cv::Point(pointX, pointY));

            // Draw point
            cv::circle(frame, cv::Point(pointX, pointY), 3, cv::Scalar(0, 255, 255), cv::FILLED);

            // Draw date label
            if (i < static_cast<int>(trend.timeLabels.size())) {
                std::string dateLabel = trend.timeLabels[i].substr(5); // Skip year
                cv::putText(
                    frame,
                    dateLabel,
                    cv::Point(pointX - 15, y + graphHeight + 15),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.4,
                    cv::Scalar(200, 200, 200),
                    1
                );
            }
        }

        // Draw lines connecting points
        for (size_t i = 1; i < points.size(); i++) {
            cv::line(frame, points[i-1], points[i], cv::Scalar(0, 255, 255), 2);
        }

        // Draw trend direction
        std::string trendText = trend.increasing ?
            "Trend: Increasing ↑" : "Trend: Decreasing ↓";

        cv::Scalar trendColor = trend.increasing ?
            cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        cv::putText(
            frame,
            trendText,
            cv::Point(x + graphWidth - 150, y + 15),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            trendColor,
            1
        );
    } else {
        // No data message
        cv::putText(
            frame,
            "No trend data available",
            cv::Point(x + 10, y + graphHeight / 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(200, 200, 200),
            1
        );
    }
}

void StatsVisualizer::renderWasteByMeal(cv::Mat& frame, int x, int y) {
    // Get waste by meal data
    auto wasteByMeal = m_analyzer->getWasteByMeal();

    // Render title
    cv::putText(
        frame,
        "Waste by Meal Period",
        cv::Point(x, y + 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(200, 200, 200),
        1
    );

    y += 40;

    if (wasteByMeal.empty()) {
        cv::putText(
            frame,
            "No meal period data available",
            cv::Point(x, y + 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(200, 200, 200),
            1
        );
        return;
    }

    // Calculate total for percentage
    float total = 0.0f;
    for (const auto& [meal, weight] : wasteByMeal) {
        total += weight;
    }

    // Avoid division by zero
    if (total == 0.0f) {
        total = 1.0f;
    }

    // Draw pie chart
    int radius = 50;
    cv::Point center(x + radius + 10, y + radius + 10);

    float startAngle = 0.0f;
    int i = 0;

    // Colors for different meal periods
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // Red
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(0, 0, 255),    // Blue
        cv::Scalar(255, 255, 0),  // Yellow
        cv::Scalar(255, 0, 255)   // Magenta
    };

    // Draw each meal segment
    for (const auto& [meal, weight] : wasteByMeal) {
        float percentage = weight / total;
        float sweepAngle = percentage * 360.0f;

        // Select color
        cv::Scalar color = colors[i % colors.size()];

        // Draw pie segment
        cv::ellipse(
            frame,
            center,
            cv::Size(radius, radius),
            0,
            startAngle,
            startAngle + sweepAngle,
            color,
            cv::FILLED
        );

        // Calculate label position
        float midAngle = startAngle + sweepAngle / 2;
        float radians = midAngle * CV_PI / 180.0f;
        int labelX = static_cast<int>(center.x + (radius + 20) * std::cos(radians));
        int labelY = static_cast<int>(center.y + (radius + 20) * std::sin(radians));

        // Draw label
        std::stringstream ss;
        ss << meal << ": " << std::fixed << std::setprecision(1) << (percentage * 100) << "%";

        cv::putText(
            frame,
            ss.str(),
            cv::Point(labelX + radius + 20, labelY + i * 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        );

        startAngle += sweepAngle;
        i++;
    }
}

void StatsVisualizer::renderInsights(cv::Mat& frame, int x, int y) {
    // Render title
    cv::putText(
        frame,
        "Key Insights",
        cv::Point(x, y + 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(200, 200, 200),
        1
    );

    y += 40;

    // Render each insight
    for (size_t i = 0; i < m_insights.size(); i++) {
        cv::putText(
            frame,
            "• " + m_insights[i],
            cv::Point(x, y + i * 25),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(200, 200, 200),
            1
        );
    }
}

//
// ControlPanel Implementation
//

ControlPanel::ControlPanel(std::shared_ptr<Utils::ConfigLoader> config,
                         std::shared_ptr<Training::ModelTrainer> trainer)
    : m_config(config),
      m_trainer(trainer),
      m_trainingInProgress(false) {

    initializeButtons();
}

void ControlPanel::initializeButtons() {
    // Add buttons with their callbacks
    m_buttons.push_back({
        cv::Rect(10, 10, 150, 30),
        "Start Training",
        [this]() {
            if (!m_trainingInProgress && m_trainer) {
                // Start training in a separate thread
                std::thread([this]() {
                    setTrainingInProgress(true);
                    m_trainer->trainModel();
                    setTrainingInProgress(false);
                }).detach();
            }
        },
        true
    });

    m_buttons.push_back({
        cv::Rect(10, 50, 150, 30),
        "Export Statistics",
        [this]() {
            // Export statistics to CSV/JSON
            std::cout << "Exporting statistics..." << std::endl;
        },
        true
    });

    m_buttons.push_back({
        cv::Rect(10, 90, 150, 30),
        "Settings",
        [this]() {
            // Open settings
            std::cout << "Opening settings..." << std::endl;
        },
        true
    });
}

void ControlPanel::render(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Draw panel background
    cv::rectangle(
        frame,
        cv::Rect(0, 0, 170, frame.rows),
        cv::Scalar(40, 40, 40),
        cv::FILLED
    );

    // Render buttons
    renderButtons(frame);

    // Render training status if in progress
    if (m_trainingInProgress) {
        cv::putText(
            frame,
            "Training in progress...",
            cv::Point(10, 140),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 255),
            1
        );
    }
}

void ControlPanel::renderButtons(cv::Mat& frame) {
    for (const auto& button : m_buttons) {
        // Draw button background
        cv::Scalar bgColor = button.enabled ?
            cv::Scalar(70, 70, 70) : cv::Scalar(50, 50, 50);

        cv::rectangle(
            frame,
            button.region,
            bgColor,
            cv::FILLED
        );

        // Draw button border
        cv::rectangle(
            frame,
            button.region,
            cv::Scalar(100, 100, 100),
            1
        );

        // Draw button label
        cv::putText(
            frame,
            button.label,
            cv::Point(button.region.x + 10, button.region.y + 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            button.enabled ? cv::Scalar(255, 255, 255) : cv::Scalar(150, 150, 150),
            1
        );
    }
}

void ControlPanel::update() {
    // Nothing to update here
}

void ControlPanel::handleMouseEvent(int event, int x, int y) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (event == cv::EVENT_LBUTTONDOWN) {
        for (auto& button : m_buttons) {
            if (button.enabled && button.region.contains(cv::Point(x, y))) {
                if (button.callback) {
                    button.callback();
                }
                break;
            }
        }
    }
}

void ControlPanel::setTrainingInProgress(bool inProgress) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_trainingInProgress = inProgress;

    // Disable training button when training is in progress
    for (auto& button : m_buttons) {
        if (button.label == "Start Training") {
            button.enabled = !inProgress;
            break;
        }
    }
}

bool ControlPanel::isTrainingInProgress() const {
    return m_trainingInProgress;
}

//
// UserInterface Implementation
//

UserInterface::UserInterface(std::shared_ptr<Camera::CameraManager> cameraManager,
                           std::shared_ptr<Detection::FoodDetector> detector,
                           std::shared_ptr<Analysis::StatsAnalyzer> analyzer,
                           std::shared_ptr<Training::ModelTrainer> trainer,
                           const Utils::ConfigLoader& config)
    : m_cameraManager(cameraManager),
      m_detector(detector),
      m_analyzer(analyzer),
      m_trainer(trainer),
      m_config(std::make_shared<Utils::ConfigLoader>(config)),
      m_running(false),
      m_currentMode(Mode::LIVE_VIEW) {

    // Create UI elements
    m_detectionVisualizer = std::make_unique<DetectionVisualizer>(detector);
    m_statsVisualizer = std::make_unique<StatsVisualizer>(analyzer);
    m_controlPanel = std::make_unique<ControlPanel>(m_config, trainer);
}

UserInterface::~UserInterface() {
    stop();
}

void UserInterface::start() {
    if (m_running) {
        return; // Already running
    }

    m_running = true;

    // Create the main window
    createMainWindow();

    std::cout << "User interface started" << std::endl;
}

void UserInterface::stop() {
    if (!m_running) {
        return; // Already stopped
    }

    m_running = false;

    // Destroy the main window
    destroyMainWindow();

    std::cout << "User interface stopped" << std::endl;
}

bool UserInterface::isRunning() const {
    return m_running;
}

void UserInterface::updateFrame(const cv::Mat& frame, const Detection::DetectionResult& detections) {
    if (!m_running) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_frameMutex);

    // Make a copy of the frame
    if (frame.empty()) {
        return;
    }

    frame.copyTo(m_displayFrame);

    // Update detection visualizer
    m_detectionVisualizer->setDetections(detections);

    // Render UI on the display frame
    renderUI();

    // Show the frame
    cv::imshow(WINDOW_NAME, m_displayFrame);
}

void UserInterface::processEvents() {
    // Process OpenCV window events
    int key = cv::waitKey(1);

    // Handle keyboard shortcuts
    switch (key) {
        case 27: // ESC key
            stop();
            break;

        case '1':
            setMode(Mode::LIVE_VIEW);
            break;

        case '2':
            setMode(Mode::STATISTICS);
            break;

        case '3':
            setMode(Mode::TRAINING);
            break;

        case '4':
            setMode(Mode::SETTINGS);
            break;

        case 's':
            // Save screenshot
            if (!m_displayFrame.empty()) {
                std::string filename = "screenshot_" + std::to_string(time(nullptr)) + ".jpg";
                cv::imwrite(filename, m_displayFrame);
                std::cout << "Screenshot saved to " << filename << std::endl;
            }
            break;
    }
}

void UserInterface::setMode(Mode mode) {
    m_currentMode = mode;

    // Update the UI based on the new mode
    std::cout << "Switching to mode: ";

    switch (mode) {
        case Mode::LIVE_VIEW:
            std::cout << "Live View" << std::endl;
            break;

        case Mode::STATISTICS:
            std::cout << "Statistics" << std::endl;
            // Make sure stats are up to date
            m_statsVisualizer->update();
            break;

        case Mode::TRAINING:
            std::cout << "Training" << std::endl;
            break;

        case Mode::SETTINGS:
            std::cout << "Settings" << std::endl;
            break;
    }
}

UserInterface::Mode UserInterface::getMode() const {
    return m_currentMode;
}

void UserInterface::renderUI() {
    // Skip if display frame is empty
    if (m_displayFrame.empty()) {
        return;
    }

    // Render different UI elements based on current mode
    switch (m_currentMode) {
        case Mode::LIVE_VIEW:
            // Render detection visualizer
            m_detectionVisualizer->render(m_displayFrame);

            // Add mode indicator
            cv::putText(
                m_displayFrame,
                "Mode: Live View (Press 2 for Statistics)",
                cv::Point(10, m_displayFrame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1
            );
            break;

        case Mode::STATISTICS:
            // Render statistics visualizer
            m_statsVisualizer->render(m_displayFrame);

            // Add mode indicator
            cv::putText(
                m_displayFrame,
                "Mode: Statistics (Press 1 for Live View)",
                cv::Point(10, m_displayFrame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1
            );
            break;

        case Mode::TRAINING:
            // Add a training background
            cv::rectangle(
                m_displayFrame,
                cv::Rect(0, 0, m_displayFrame.cols, m_displayFrame.rows),
                cv::Scalar(40, 40, 40),
                cv::FILLED
            );

            // Add training title
            cv::putText(
                m_displayFrame,
                "Model Training",
                cv::Point(m_displayFrame.cols / 2 - 100, 50),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(255, 255, 255),
                2
            );

            // Display training status
            std::string statusText = m_controlPanel->isTrainingInProgress() ?
                "Training in progress..." : "Ready to train";

            cv::putText(
                m_displayFrame,
                statusText,
                cv::Point(m_displayFrame.cols / 2 - 100, 100),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(200, 200, 200),
                1
            );

            // Add mode indicator
            cv::putText(
                m_displayFrame,
                "Mode: Training (Press 1 for Live View)",
                cv::Point(10, m_displayFrame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1
            );
            break;

        case Mode::SETTINGS:
            // Add a settings background
            cv::rectangle(
                m_displayFrame,
                cv::Rect(0, 0, m_displayFrame.cols, m_displayFrame.rows),
                cv::Scalar(40, 40, 40),
                cv::FILLED
            );

            // Add settings title
            cv::putText(
                m_displayFrame,
                "Settings",
                cv::Point(m_displayFrame.cols / 2 - 50, 50),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(255, 255, 255),
                2
            );

            // Add mode indicator
            cv::putText(
                m_displayFrame,
                "Mode: Settings (Press 1 for Live View)",
                cv::Point(10, m_displayFrame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1
            );
            break;
    }

    // Always render control panel
    m_controlPanel->render(m_displayFrame);
}

void UserInterface::createMainWindow() {
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 1280, 720);

    // Set mouse callback
    cv::setMouseCallback(WINDOW_NAME, onMouse, this);

    // Create initial black frame
    m_displayFrame = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));

    // Add welcome text
    cv::putText(
        m_displayFrame,
        "Food Waste Monitor",
        cv::Point(m_displayFrame.cols / 2 - 150, m_displayFrame.rows / 2 - 20),
        cv::FONT_HERSHEY_SIMPLEX,
        1.2,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::putText(
        m_displayFrame,
        "Starting camera...",
        cv::Point(m_displayFrame.cols / 2 - 100, m_displayFrame.rows / 2 + 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(200, 200, 200),
        1
    );

    // Show initial frame
    cv::imshow(WINDOW_NAME, m_displayFrame);
    cv::waitKey(1);
}

void UserInterface::destroyMainWindow() {
    cv::destroyWindow(WINDOW_NAME);
}

void UserInterface::onMouse(int event, int x, int y, int flags, void* userdata) {
    // Forward the mouse event to the instance
    UserInterface* ui = static_cast<UserInterface*>(userdata);
    if (ui) {
        ui->handleMouseEvent(event, x, y);
    }
}

void UserInterface::handleMouseEvent(int event, int x, int y) {
    // Forward the mouse event to the appropriate UI element based on mode
    switch (m_currentMode) {
        case Mode::LIVE_VIEW:
            m_detectionVisualizer->handleMouseEvent(event, x, y);
            break;

        case Mode::STATISTICS:
            m_statsVisualizer->handleMouseEvent(event, x, y);
            break;

        case Mode::TRAINING:
        case Mode::SETTINGS:
            // No specific handlers for these modes
            break;
    }

    // Always forward to control panel
    m_controlPanel->handleMouseEvent(event, x, y);
}

} // namespace UI