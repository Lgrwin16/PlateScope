/**
* Camera Management Header
 *
 * Handles interfacing with OpenCV camera capture and frame processing
 */

#ifndef CAMERA_MANAGER_H
#define CAMERA_MANAGER_H

#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>

namespace Camera {

    class CameraManager {
    public:
        explicit CameraManager(int cameraIndex = 0);
        ~CameraManager();

        // Camera control functions
        bool start();
        void stop();
        bool isRunning() const;

        // Frame access functions
        bool hasNewFrame() const;
        cv::Mat getLatestFrame();

        // Camera settings
        bool setResolution(int width, int height);
        bool setFrameRate(int fps);
        bool setExposure(double exposure);
        bool setAutoExposure(bool enable);
        bool setWhiteBalance(double value);
        bool setAutoWhiteBalance(bool enable);

        // Camera information
        cv::Size getResolution() const;
        double getFrameRate() const;

    private:
        void captureThread();
        void processFrame(cv::Mat& frame);

        cv::VideoCapture m_camera;
        int m_cameraIndex;
        cv::Mat m_latestFrame;

        std::atomic<bool> m_running;
        std::atomic<bool> m_newFrameAvailable;

        std::thread m_captureThread;
        std::mutex m_frameMutex;

        // Frame buffer for processing
        std::queue<cv::Mat> m_frameQueue;
        std::mutex m_queueMutex;
        std::condition_variable m_queueCondition;

        // Camera properties
        int m_width;
        int m_height;
        double m_fps;
    };

} // namespace Camera

#endif // CAMERA_MANAGER_H