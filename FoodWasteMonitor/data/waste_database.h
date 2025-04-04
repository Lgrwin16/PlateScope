/**
 * Food Waste Database Management Header
 *
 * Manages storage, retrieval, and analysis of food waste data
 */

#ifndef WASTE_DATABASE_H
#define WASTE_DATABASE_H

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <atomic>
#include <functional>
#include "../detection/food_detector.h"

namespace Data {

// Time period definitions for statistics
enum class TimePeriod {
    DAY,
    WEEK,
    MONTH,
    YEAR,
    ALL_TIME
};

// Meal period definitions
enum class MealPeriod {
    BREAKFAST,
    LUNCH,
    DINNER,
    SNACK,
    UNKNOWN
};

// Structure to represent a waste entry in the database
struct WasteEntry {
    std::string foodType;         // Type of food
    float weight;                 // Weight in grams
    std::string timestamp;        // Time of detection
    float confidence;             // Detection confidence
    std::string mealPeriod;       // Breakfast, lunch, dinner, etc.
    std::string imageFilename;    // Path to saved image

    WasteEntry() : weight(0.0f), confidence(0.0f) {}
};

// Statistics summary structure
struct WasteStatistics {
    // Total waste statistics
    float totalWeight;                    // Total weight of waste
    int totalItems;                       // Total number of waste items

    // Food type statistics
    std::map<std::string, float> weightByType;     // Weight by food type
    std::map<std::string, int> countByType;        // Count by food type
    std::vector<std::string> topWastedFoods;       // Top wasted foods by weight

    // Time-related statistics
    std::map<std::string, float> weightByDay;      // Weight by day of week
    std::map<std::string, float> weightByMeal;     // Weight by meal period
    std::map<std::string, float> weightByMonth;    // Weight by month

    // Trend statistics
    std::vector<float> dailyTrend;                 // Daily waste trend
    std::vector<float> weeklyTrend;                // Weekly waste trend
    std::vector<float> monthlyTrend;               // Monthly waste trend

    // Waste reduction statistics
    float wasteSavedTotal;                         // Total waste reduction
    float wasteSavedPercentage;                    // Waste reduction percentage

    WasteStatistics() : totalWeight(0.0f), totalItems(0), wasteSavedTotal(0.0f), wasteSavedPercentage(0.0f) {}
};

class WasteDatabase {
public:
    explicit WasteDatabase(const std::string& databasePath);
    ~WasteDatabase();

    // Database operations
    bool initialize();
    bool saveToFile();
    bool loadFromFile();

    // Data addition
    void addDetection(const Detection::FoodItem& item);
    void addDetections(const Detection::DetectionResult& detections);
    void addEntry(const WasteEntry& entry);

    // Data retrieval
    std::vector<WasteEntry> getEntries(
        const std::string& foodType = "",
        const std::string& startDate = "",
        const std::string& endDate = ""
    );

    // Statistics generation
    WasteStatistics getStatistics(TimePeriod period = TimePeriod::ALL_TIME);

    // Get specific statistics
    std::vector<std::string> getTopWastedFoods(int limit = 5);
    float getTotalWasteWeight(TimePeriod period = TimePeriod::ALL_TIME);
    float getAverageWastePerDay(TimePeriod period = TimePeriod::MONTH);
    std::map<std::string, float> getWasteByType(TimePeriod period = TimePeriod::ALL_TIME);
    std::map<std::string, float> getWasteByMeal(TimePeriod period = TimePeriod::ALL_TIME);
    std::map<std::string, float> getWasteTrend(TimePeriod period = TimePeriod::MONTH);

    // Export functions
    bool exportToCSV(const std::string& filePath);
    bool exportToJSON(const std::string& filePath);

    // Utility functions
    void setMealPeriod(MealPeriod period);
    MealPeriod getCurrentMealPeriod() const;
    std::string getMealPeriodString() const;
    void saveDetectionImage(const cv::Mat& frame, const Detection::FoodItem& item, std::string& outputPath);

    // Observer pattern for database changes
    using DatabaseChangeCallback = std::function<void()>;
    void registerChangeCallback(DatabaseChangeCallback callback);

private:
    // Initialize meal periods
    void initializeMealPeriods();

    // Determine meal period from time
    MealPeriod determineMealPeriod(const std::string& timestamp) const;

    // Filter entries by date range
    std::vector<WasteEntry> filterEntriesByDate(
        const std::vector<WasteEntry>& entries,
        const std::string& startDate,
        const std::string& endDate
    );

    // Calculate statistics
    void calculateStatistics();

    // Database storage
    std::string m_databasePath;
    std::vector<WasteEntry> m_entries;
    WasteStatistics m_statistics;
    std::atomic<bool> m_statisticsDirty;

    // Current application state
    MealPeriod m_currentMealPeriod;

    // Meal time ranges (24-hour format)
    struct TimeRange {
        int startHour;
        int startMinute;
        int endHour;
        int endMinute;
    };
    std::map<MealPeriod, TimeRange> m_mealTimeRanges;

    // Thread safety
    mutable std::mutex m_databaseMutex;

    // Observer pattern
    std::vector<DatabaseChangeCallback> m_changeCallbacks;

    // Notification of changes
    void notifyDatabaseChanged();
};