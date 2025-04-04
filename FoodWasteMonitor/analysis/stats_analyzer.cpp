/**
 * Statistical Analysis Implementation
 */

#include "stats_analyzer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace Analysis {

StatsAnalyzer::StatsAnalyzer(std::shared_ptr<Data::WasteDatabase> database)
    : m_database(database),
      m_insightsDirty(true) {

    // Initialize with current stats
    updateStats();
}

void StatsAnalyzer::updateStats() {
    // Get latest statistics from the database
    m_currentStats = m_database->getStatistics();

    // Mark insights as dirty (need to be recalculated)
    m_insightsDirty = true;

    // Update trends
    m_dailyTrend = analyzeDailyTrend();

    // Update prediction model
    if (!m_dailyTrend.values.empty()) {
        std::vector<float> xValues(m_dailyTrend.values.size());
        std::iota(xValues.begin(), xValues.end(), 0);

        performLinearRegression(xValues, m_dailyTrend.values,
                               m_wastePredictionModel.intercept,
                               m_wastePredictionModel.slope,
                               m_wastePredictionModel.rSquared);
    }
}

TrendData StatsAnalyzer::analyzeDailyTrend(int days) {
    TrendData trend;

    // Get waste data for specified time period
    auto wasteTrend = m_database->getWasteTrend(days <= 7 ? Data::TimePeriod::WEEK :
                                               (days <= 30 ? Data::TimePeriod::MONTH : Data::TimePeriod::YEAR));

    // Sort the data by date
    std::vector<std::pair<std::string, float>> sortedData(wasteTrend.begin(), wasteTrend.end());
    std::sort(sortedData.begin(), sortedData.end());

    // Limit to requested number of days
    if (sortedData.size() > static_cast<size_t>(days)) {
        sortedData.erase(sortedData.begin(), sortedData.end() - days);
    }

    // Extract data into trend vectors
    trend.timeLabels.clear();
    trend.values.clear();

    for (const auto& [date, weight] : sortedData) {
        trend.timeLabels.push_back(date);
        trend.values.push_back(weight);
    }

    // Calculate trend percentage
    trend.changePercentage = calculateTrendPercentage(trend.values);
    trend.increasing = trend.changePercentage > 0;

    return trend;
}

TrendData StatsAnalyzer::analyzeFoodTypeTrend(const std::string& foodType, int days) {
    TrendData trend;

    // Get entries for the food type
    auto entries = m_database->getEntries(foodType);

    if (entries.empty()) {
        return trend;
    }

    // Group by date
    std::map<std::string, float> dailyWeights;

    for (const auto& entry : entries) {
        // Extract date part from timestamp
        std::string dateStr = entry.timestamp.substr(0, 10);
        dailyWeights[dateStr] += entry.weight;
    }

    // Sort by date
    std::vector<std::pair<std::string, float>> sortedData(dailyWeights.begin(), dailyWeights.end());
    std::sort(sortedData.begin(), sortedData.end());

    // Limit to requested number of days
    if (sortedData.size() > static_cast<size_t>(days)) {
        sortedData.erase(sortedData.begin(), sortedData.end() - days);
    }

    // Extract data into trend vectors
    for (const auto& [date, weight] : sortedData) {
        trend.timeLabels.push_back(date);
        trend.values.push_back(weight);
    }

    // Calculate trend percentage
    trend.changePercentage = calculateTrendPercentage(trend.values);
    trend.increasing = trend.changePercentage > 0;

    // Cache the result
    m_foodTypeTrends[foodType] = trend;

    return trend;
}

TrendData StatsAnalyzer::analyzeMealPeriodTrend(const std::string& mealPeriod, int days) {
    TrendData trend;

    // Get all entries
    auto entries = m_database->getEntries();

    if (entries.empty()) {
        return trend;
    }

    // Group by date for the specific meal period
    std::map<std::string, float> dailyWeights;

    for (const auto& entry : entries) {
        if (entry.mealPeriod == mealPeriod) {
            // Extract date part from timestamp
            std::string dateStr = entry.timestamp.substr(0, 10);
            dailyWeights[dateStr] += entry.weight;
        }
    }

    // Sort by date
    std::vector<std::pair<std::string, float>> sortedData(dailyWeights.begin(), dailyWeights.end());
    std::sort(sortedData.begin(), sortedData.end());

    // Limit to requested number of days
    if (sortedData.size() > static_cast<size_t>(days)) {
        sortedData.erase(sortedData.begin(), sortedData.end() - days);
    }

    // Extract data into trend vectors
    for (const auto& [date, weight] : sortedData) {
        trend.timeLabels.push_back(date);
        trend.values.push_back(weight);
    }

    // Calculate trend percentage
    trend.changePercentage = calculateTrendPercentage(trend.values);
    trend.increasing = trend.changePercentage > 0;

    // Cache the result
    m_mealPeriodTrends[mealPeriod] = trend;

    return trend;
}

float StatsAnalyzer::calculateTrendPercentage(const std::vector<float>& values) {
    if (values.size() < 2) {
        return 0.0f;
    }

    // Use linear regression to determine trend
    std::vector<float> xValues(values.size());
    std::iota(xValues.begin(), xValues.end(), 0);

    float intercept, slope, rSquared;
    performLinearRegression(xValues, values, intercept, slope, rSquared);

    // Calculate percentage change over the period
    float startValue = values.front();
    float endValue = values.back();

    if (startValue == 0) {
        return 0.0f;  // Avoid division by zero
    }

    return ((endValue - startValue) / startValue) * 100.0f;
}

void StatsAnalyzer::performLinearRegression(const std::vector<float>& xValues, const std::vector<float>& yValues,
                               float& intercept, float& slope, float& rSquared) {
    if (xValues.size() != yValues.size() || xValues.size() < 2) {
        // Can't perform regression with mismatched data or too few points
        intercept = 0.0f;
        slope = 0.0f;
        rSquared = 0.0f;
        return;
    }

    int n = static_cast<int>(xValues.size());

    // Calculate means
    float meanX = std::accumulate(xValues.begin(), xValues.end(), 0.0f) / n;
    float meanY = std::accumulate(yValues.begin(), yValues.end(), 0.0f) / n;

    // Calculate sum of squares
    float sumXY = 0.0f, sumXX = 0.0f, sumYY = 0.0f;

    for (int i = 0; i < n; i++) {
        float xDiff = xValues[i] - meanX;
        float yDiff = yValues[i] - meanY;

        sumXY += xDiff * yDiff;
        sumXX += xDiff * xDiff;
        sumYY += yDiff * yDiff;
    }

    // Check for division by zero
    if (sumXX < 1e-9) {
        intercept = meanY;
        slope = 0.0f;
        rSquared = 0.0f;
        return;
    }

    // Calculate slope and intercept
    slope = sumXY / sumXX;
    intercept = meanY - slope * meanX;

    // Calculate R-squared
    float ssr = 0.0f;  // Sum of squared residuals
    float sst = sumYY; // Total sum of squares

    for (int i = 0; i < n; i++) {
        float predicted = intercept + slope * xValues[i];
        float residual = yValues[i] - predicted;
        ssr += residual * residual;
    }

    if (sst < 1e-9) {
        rSquared = 0.0f;
    } else {
        rSquared = 1.0f - (ssr / sst);
    }
}

PredictionModel StatsAnalyzer::createPredictionModel(const std::vector<float>& data) {
    PredictionModel model;

    if (data.size() < 2) {
        return model; // Not enough data for prediction
    }

    // Create x values (0, 1, 2, ..., n-1)
    std::vector<float> xValues(data.size());
    std::iota(xValues.begin(), xValues.end(), 0);

    // Perform linear regression
    performLinearRegression(xValues, data, model.intercept, model.slope, model.rSquared);

    return model;
}

float StatsAnalyzer::predictFutureWaste(int daysInFuture) {
    if (daysInFuture < 0) {
        return 0.0f;
    }

    // Use the cached prediction model
    if (std::abs(m_wastePredictionModel.rSquared) < 0.1f) {
        // If model has poor fit, update it
        updateStats();
    }

    // Calculate predicted value using linear model
    // y = intercept + slope * x
    // where x is the day in the future (current day + daysInFuture)
    float currentDay = static_cast<float>(m_dailyTrend.values.size() - 1);
    float futureDay = currentDay + daysInFuture;

    float predictedWaste = m_wastePredictionModel.intercept + m_wastePredictionModel.slope * futureDay;

    // Ensure prediction is not negative
    return std::max(0.0f, predictedWaste);
}

std::vector<WasteRecommendation> StatsAnalyzer::generateRecommendations(int limit) {
    std::vector<WasteRecommendation> recommendations;

    // Get top wasted foods
    auto topFoods = m_database->getTopWastedFoods(5);

    // Get waste by meal period
    auto wasteByMeal = m_database->getWasteByMeal();

    // Find top waste combinations (food type × meal period)
    std::map<std::pair<std::string, std::string>, float> combinedWaste;

    // Get all entries to analyze
    auto entries = m_database->getEntries();

    for (const auto& entry : entries) {
        // Key is food type + meal period
        std::pair<std::string, std::string> key = {entry.foodType, entry.mealPeriod};
        combinedWaste[key] += entry.weight;
    }

    // Sort by waste weight
    std::vector<std::pair<std::pair<std::string, std::string>, float>> sortedCombinations(
        combinedWaste.begin(), combinedWaste.end());

    std::sort(sortedCombinations.begin(), sortedCombinations.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Generate recommendations for the top wasting combinations
    for (size_t i = 0; i < std::min(static_cast<size_t>(limit), sortedCombinations.size()); i++) {
        const auto& [key, weight] = sortedCombinations[i];
        const auto& [foodType, mealPeriod] = key;

        WasteRecommendation rec;
        rec.foodType = foodType;
        rec.mealPeriod = mealPeriod;

        // Calculate potential savings (assume 30% reduction is possible)
        rec.potentialSavings = weight * 0.3f;

        // Generate a recommendation message based on the combination
        std::stringstream ss;
        ss << "Consider reducing portion sizes for " << foodType << " during " << mealPeriod
           << ". Current waste is approximately " << std::fixed << std::setprecision(1) << weight
           << "g, with potential savings of " << rec.potentialSavings << "g per day.";

        rec.recommendation = ss.str();

        recommendations.push_back(rec);
    }

    return recommendations;
}

std::vector<std::string> StatsAnalyzer::getInsights() {
    // Regenerate insights if needed
    if (m_insightsDirty || m_insights.empty()) {
        m_insights.clear();

        // Get current statistics
        auto stats = m_currentStats;

        // Add basic insights about total waste
        std::stringstream ss;
        ss << "Total food waste recorded: " << std::fixed << std::setprecision(1)
           << stats.totalWeight << "g across " << stats.totalItems << " items.";
        m_insights.push_back(ss.str());

        // Add insights about top wasted foods
        if (!stats.topWastedFoods.empty()) {
            ss.str("");
            ss << "The most wasted food is " << stats.topWastedFoods[0] << " at "
               << std::fixed << std::setprecision(1) << stats.weightByType[stats.topWastedFoods[0]] << "g.";
            m_insights.push_back(ss.str());
        }

        // Add trend insight
        if (!m_dailyTrend.values.empty() && m_dailyTrend.values.size() > 1) {
            ss.str("");
            ss << "Waste is " << (m_dailyTrend.increasing ? "increasing" : "decreasing")
               << " by " << std::abs(m_dailyTrend.changePercentage) << "% over the last "
               << m_dailyTrend.values.size() << " days.";
            m_insights.push_back(ss.str());
        }

        // Add meal period insight
        if (!stats.weightByMeal.empty()) {
            // Find meal with highest waste
            auto maxMeal = std::max_element(stats.weightByMeal.begin(), stats.weightByMeal.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; });

            ss.str("");
            ss << "The meal period with highest waste is " << maxMeal->first << " at "
               << std::fixed << std::setprecision(1) << maxMeal->second << "g.";
            m_insights.push_back(ss.str());
        }

        // Add day of week insight
        if (!stats.weightByDay.empty()) {
            // Find day with highest waste
            auto maxDay = std::max_element(stats.weightByDay.begin(), stats.weightByDay.end(),
                                         [](const auto& a, const auto& b) { return a.second < b.second; });

            ss.str("");
            ss << "The day with highest waste is " << maxDay->first << " at "
               << std::fixed << std::setprecision(1) << maxDay->second << "g.";
            m_insights.push_back(ss.str());
        }

        // Add waste reduction insight
        if (stats.wasteSavedTotal > 0) {
            ss.str("");
            ss << "Waste has been reduced by " << std::fixed << std::setprecision(1)
               << stats.wasteSavedTotal << "g (" << stats.wasteSavedPercentage << "%) compared to the previous period.";
            m_insights.push_back(ss.str());
        }

        // Add prediction insight
        float nextWeekPrediction = predictFutureWaste(7);
        ss.str("");
        ss << "Predicted waste for next week: " << std::fixed << std::setprecision(1) << nextWeekPrediction << "g.";
        m_insights.push_back(ss.str());

        m_insightsDirty = false;
    }

    return m_insights;
}

float StatsAnalyzer::calculateWasteCost(float pricePerKg) {
    // Convert grams to kg and multiply by price
    return (m_currentStats.totalWeight / 1000.0f) * pricePerKg;
}

float StatsAnalyzer::calculatePotentialSavings(int days, float pricePerKg) {
    // Assume a 30% reduction is possible
    float potentialDailyReduction = m_database->getAverageWastePerDay() * 0.3f;

    // Calculate total potential savings over the specified period
    float potentialWeightSavings = potentialDailyReduction * days;

    // Convert to cost savings
    return (potentialWeightSavings / 1000.0f) * pricePerKg;
}

float StatsAnalyzer::calculateCO2Impact(float kgCO2PerKgFood) {
    // Convert grams to kg and multiply by CO2 impact factor
    return (m_currentStats.totalWeight / 1000.0f) * kgCO2PerKgFood;
}

float StatsAnalyzer::calculateWaterImpact(float litersPerKgFood) {
    // Convert grams to kg and multiply by water impact factor
    return (m_currentStats.totalWeight / 1000.0f) * litersPerKgFood;
}

std::map<std::string, float> StatsAnalyzer::identifyOutliers(const std::map<std::string, float>& data, float threshold) {
    std::map<std::string, float> outliers;

    if (data.empty()) {
        return outliers;
    }

    // Calculate mean and standard deviation
    std::vector<float> values;
    for (const auto& [key, value] : data) {
        values.push_back(value);
    }

    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    float mean = sum / values.size();

    float varianceSum = 0.0f;
    for (float value : values) {
        float diff = value - mean;
        varianceSum += diff * diff;
    }
    float stdDev = std::sqrt(varianceSum / values.size());

    // Identify outliers (values more than threshold standard deviations from mean)
    for (const auto& [key, value] : data) {
        float zScore = std::abs(value - mean) / stdDev;
        if (zScore > threshold) {
            outliers[key] = value;
        }
    }

    return outliers;
}

std::vector<std::string> StatsAnalyzer::findCorrelations() {
    std::vector<std::string> results;

    // This is a simplified implementation. In a complete system,
    // this would perform correlation analysis between different
    // factors such as day of week, meal period, and waste amounts.

    // Get day of week pattern
    auto dayPattern = calculateDayOfWeekPattern();

    // Get meal period pattern
    auto mealPattern = m_currentStats.weightByMeal;

    // Find days with significantly higher waste
    auto dayOutliers = identifyOutliers(dayPattern);

    for (const auto& [day, weight] : dayOutliers) {
        std::stringstream ss;
        ss << "Correlation found: " << day << " consistently has higher waste ("
           << std::fixed << std::setprecision(1) << weight << "g on average).";
        results.push_back(ss.str());
    }

    // Find meal periods with significantly higher waste
    auto mealOutliers = identifyOutliers(mealPattern);

    for (const auto& [meal, weight] : mealOutliers) {
        std::stringstream ss;
        ss << "Correlation found: " << meal << " consistently has higher waste ("
           << std::fixed << std::setprecision(1) << weight << "g on average).";
        results.push_back(ss.str());
    }

    return results;
}

std::map<std::string, float> StatsAnalyzer::calculateDayOfWeekPattern() {
    std::map<std::string, float> pattern;
    std::map<std::string, int> dayCounts;

    // Array of day names
    const std::array<std::string, 7> dayNames = {
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
    };

    // Initialize pattern with zeros
    for (const auto& day : dayNames) {
        pattern[day] = 0.0f;
        dayCounts[day] = 0;
    }

    // Get all entries
    auto entries = m_database->getEntries();

    for (const auto& entry : entries) {
        // Parse timestamp to get day of week
        std::tm tm = {};
        std::istringstream ss(entry.timestamp);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

        if (!ss.fail()) {
            // Get day of week (0 = Sunday, 6 = Saturday)
            int dayIndex = tm.tm_wday;
            std::string dayName = dayNames[dayIndex];

            pattern[dayName] += entry.weight;
            dayCounts[dayName]++;
        }
    }

    // Calculate average waste per day
    for (auto& [day, weight] : pattern) {
        if (dayCounts[day] > 0) {
            weight /= dayCounts[day];
        }
    }

    return pattern;
}

std::map<std::string, float> StatsAnalyzer::calculateMonthlyPattern() {
    std::map<std::string, float> pattern;
    std::map<std::string, int> monthCounts;

    // Array of month names
    const std::array<std::string, 12> monthNames = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    };

    // Initialize pattern with zeros
    for (const auto& month : monthNames) {
        pattern[month] = 0.0f;
        monthCounts[month] = 0;
    }

    // Get all entries
    auto entries = m_database->getEntries();

    for (const auto& entry : entries) {
        // Parse timestamp to get month
        std::tm tm = {};
        std::istringstream ss(entry.timestamp);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

        if (!ss.fail()) {
            // Get month (0 = January, 11 = December)
            int monthIndex = tm.tm_mon;
            std::string monthName = monthNames[monthIndex];

            pattern[monthName] += entry.weight;
            monthCounts[monthName]++;
        }
    }

    // Calculate average waste per month
    for (auto& [month, weight] : pattern) {
        if (monthCounts[month] > 0) {
            weight /= monthCounts[month];
        }
    }

    return pattern;
}

std::vector<float> StatsAnalyzer::calculateMovingAverage(const std::vector<float>& values, int windowSize) {
    if (values.empty() || windowSize <= 0 || windowSize > static_cast<int>(values.size())) {
        return values;  // Return original if can't compute moving average
    }

    std::vector<float> result(values.size());

    for (size_t i = 0; i < values.size(); i++) {
        float sum = 0.0f;
        int count = 0;

        // Calculate average of window centered at current point
        for (int j = -windowSize / 2; j <= windowSize / 2; j++) {
            int idx = static_cast<int>(i) + j;
            if (idx >= 0 && idx < static_cast<int>(values.size())) {
                sum += values[idx];
                count++;
            }
        }

        result[i] = (count > 0) ? sum / count : values[i];
    }

    return result;
}