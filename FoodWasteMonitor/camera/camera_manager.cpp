/**
 * Camera Management Implementation
 */

#include "camera_manager.h"
#include <iostream>

namespace Camera {

CameraManager::CameraManager(int cameraIndex)
    : m_cameraIndex(cameraIndex),
      m_running(false),
      m_newFrameAvailable(false),
      m_width(1280),
      m_height(720),
      m_fps(30.0) {
}

CameraManager::~CameraManager() {
    stop();
}

bool CameraManager::start() {
    if (m_running) {
        return true; // Already running
    }

    // Open the camera
    if (!m_camera.open(m_cameraIndex)) {
        std::cerr << "Error: Could not open camera with index " << m_cameraIndex << std::endl;
        return false;
    }

    // Set camera properties
    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, m_width);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, m_height);
    m_camera.set(cv::CAP_PROP_FPS, m_fps);

    // Check if camera is opened successfully
    if (!m_camera.isOpened()) {
        std::cerr << "Error: Camera failed to initialize properly." << std::endl;
        return false;
    }

    // Start the capture thread
    m_running = true;
    m_captureThread = std::thread(&CameraManager::captureThread, this);

    std::cout << "Camera started successfully at " << m_width << "x" << m_height
              << " @ " << m_fps << " fps" << std::endl;
    return true;
}

void CameraManager::stop() {
    if (!m_running) {
        return; // Already stopped
    }

    // Signal thread to stop and wait for it
    m_running = false;

    // Notify waiting threads
    m_queueCondition.notify_all();

    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }

    // Release camera resources
    m_camera.release();

    std::cout << "Camera stopped." << std::endl;
}

bool CameraManager::isRunning() const {
    return m_running;
}

bool CameraManager::hasNewFrame() const {
    return m_newFrameAvailable;
}

cv::Mat CameraManager::getLatestFrame() {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_newFrameAvailable = false;
    return m_latestFrame.clone();
}

void CameraManager::captureThread() {
    cv::Mat frame;

    while (m_running) {
        // Capture a new frame
        bool success = m_camera.read(frame);

        if (!success) {
            std::cerr << "Warning: Failed to read frame from camera" << std::endl;
            // Small delay to prevent CPU hogging in case of failure
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        // Pre-process the frame if needed
        processFrame(frame);

        // Update the latest frame
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            m_latestFrame = frame.clone();
            m_newFrameAvailable = true;
        }

        // Add to processing queue
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_frameQueue.push(frame.clone());
        }
        m_queueCondition.notify_one();
    }
}

void CameraManager::processFrame(cv::Mat& frame) {
    // Apply basic pre-processing
    // This could include resizing, color conversion, noise reduction, etc.

    // Ensure consistent size
    if (frame.size().width != m_width || frame.size().height != m_height) {
        cv::resize(frame, frame, cv::Size(m_width, m_height));
    }

    // Optional: Apply noise reduction
    // cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);
}

bool CameraManager::setResolution(int width, int height) {
    if (!m_camera.isOpened()) {
        m_width = width;
        m_height = height;
        return true;
    }

    bool wasRunning = m_running;
    if (wasRunning) {
        stop();
    }

    m_width = width;
    m_height = height;

    if (wasRunning) {
        return start();
    }
    return true;
}

bool CameraManager::setFrameRate(int fps) {
    m_fps = static_cast<double>(fps);
    if (m_camera.isOpened()) {
        return m_camera.set(cv::CAP_PROP_FPS, m_fps);
    }
    return true;
}

bool CameraManager::setExposure(double exposure) {
    if (m_camera.isOpened()) {
        return m_camera.set(cv::CAP_PROP_EXPOSURE, exposure);
    }
    return false;
}

bool CameraManager::setAutoExposure(bool enable) {
    if (m_camera.isOpened()) {
        return m_camera.set(cv::CAP_PROP_AUTO_EXPOSURE, enable ? 1.0 : 0.0);
    }
    return false;
}

bool CameraManager::setWhiteBalance(double value) {
    if (m_camera.isOpened()) {
        return m_camera.set(cv::CAP_PROP_WB_TEMPERATURE, value);
    }
    return false;
}

bool CameraManager::setAutoWhiteBalance(bool enable) {
    if (m_camera.isOpened()) {
        return m_camera.set(cv::CAP_PROP_AUTO_WB, enable ? 1.0 : 0.0);
    }
    return false;
}

cv::Size CameraManager::getResolution() const {
    return cv::Size(m_width, m_height);
}

double CameraManager::getFrameRate() const {
    return m_fps;
}

} // namespace Camera