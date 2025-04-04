/**
 * Statistical Analysis Module Header
 *
 * Performs statistical analysis on food waste data to provide insights
 */

#ifndef STATS_ANALYZER_H
#define STATS_ANALYZER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "../data/waste_database.h"

namespace Analysis {

// Structure for trend analysis
struct TrendData {
    std::vector<std::string> timeLabels;
    std::vector<float> values;
    float changePercentage;
    bool increasing;

    TrendData() : changePercentage(0.0f), increasing(false) {}
};

// Structure for a prediction model
struct PredictionModel {
    float intercept;
    float slope;
    float rSquared;

    PredictionModel() : intercept(0.0f), slope(0.0f), rSquared(0.0f) {}
};

// Structure for holding recommendations
struct WasteRecommendation {
    std::string foodType;
    std::string mealPeriod;
    std::string recommendation;
    float potentialSavings;

    WasteRecommendation() : potentialSavings(0.0f) {}
};

class StatsAnalyzer {
public:
    explicit StatsAnalyzer(std::shared_ptr<Data::WasteDatabase> database);

    // Core analysis functions
    void updateStats();

    // Trend analysis
    TrendData analyzeDailyTrend(int days = 30);
    TrendData analyzeFoodTypeTrend(const std::string& foodType, int days = 30);
    TrendData analyzeMealPeriodTrend(const std::string& mealPeriod, int days = 30);

    // Prediction functions
    PredictionModel createPredictionModel(const std::vector<float>& data);
    float predictFutureWaste(int daysInFuture);

    // Insight generation
    std::vector<WasteRecommendation> generateRecommendations(int limit = 3);
    std::vector<std::string> getInsights();

    // Cost analysis
    float calculateWasteCost(float pricePerKg = 5.0f);
    float calculatePotentialSavings(int days = 30, float pricePerKg = 5.0f);

    // Environmental impact
    float calculateCO2Impact(float kgCO2PerKgFood = 2.5f);
    float calculateWaterImpact(float litersPerKgFood = 1000.0f);

private:
    // Trend calculation helpers
    float calculateTrendPercentage(const std::vector<float>& values);
    std::vector<float> calculateMovingAverage(const std::vector<float>& values, int windowSize = 3);

    // Linear regression for predictions
    void performLinearRegression(const std::vector<float>& xValues, const std::vector<float>& yValues,
                                float& intercept, float& slope, float& rSquared);

    // Helper functions for insight generation
    std::map<std::string, float> identifyOutliers(const std::map<std::string, float>& data, float threshold = 1.5f);
    std::vector<std::string> findCorrelations();

    // Calculate weekly, monthly patterns
    std::map<std::string, float> calculateDayOfWeekPattern();
    std::map<std::string, float> calculateMonthlyPattern();

    // Database reference
    std::shared_ptr<Data::WasteDatabase> m_database;

    // Cached stats
    Data::WasteStatistics m_currentStats;

    // Cached trends
    TrendData m_dailyTrend;
    std::map<std::string, TrendData> m_foodTypeTrends;
    std::map<std::string, TrendData> m_mealPeriodTrends;

    // Cached prediction model
    PredictionModel m_wastePredictionModel;

    // Insights cache
    std::vector<std::string> m_insights;
    bool m_insightsDirty;
};

} // namespace Analysis

#endif // STATS_ANALYZER_H