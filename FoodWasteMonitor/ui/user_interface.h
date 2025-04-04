/**
* User Interface Module Header
 *
 * Handles user interaction, visualization, and display of results
 */

#ifndef USER_INTERFACE_H
#define USER_INTERFACE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>

#include "../camera/camera_manager.h"
#include "../detection/food_detector.h"
#include "../analysis/stats_analyzer.h"
#include "../training/model_trainer.h"
#include "../utils/config_loader.h"

namespace UI {

    // UI Element Base Class
    class UIElement {
    public:
        virtual ~UIElement() = default;
        virtual void render(cv::Mat& frame) = 0;
        virtual void update() = 0;
        virtual void handleMouseEvent(int event, int x, int y) = 0;
    };