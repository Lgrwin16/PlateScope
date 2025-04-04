/**
 * Food Waste Database Implementation
 */

#include "waste_database.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace Data {

WasteDatabase::WasteDatabase(const std::string& databasePath)
    : m_databasePath(databasePath),
      m_statisticsDirty(true),
      m_currentMealPeriod(MealPeriod::UNKNOWN) {

    initializeMealPeriods();
    initialize();
}

WasteDatabase::~WasteDatabase() {
    saveToFile();
}

bool WasteDatabase::initialize() {
    // Create directory if it doesn't exist
    fs::path dbPath(m_databasePath);
    fs::path dbDir = dbPath.parent_path();

    try {
        if (!fs::exists(dbDir)) {
            fs::create_directories(dbDir);
        }

        // Load existing data if file exists
        if (fs::exists(dbPath)) {
            return loadFromFile();
        }

        // Otherwise initialize empty database
        {
            std::lock_guard<std::mutex> lock(m_databaseMutex);
            m_entries.clear();
            m_statisticsDirty = true;
        }

        std::cout << "Initialized new database at " << m_databasePath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing database: " << e.what() << std::endl;
        return false;
    }
}

void WasteDatabase::initializeMealPeriods() {
    // Define standard meal periods (can be customized)
    m_mealTimeRanges[MealPeriod::BREAKFAST] = {6, 0, 10, 30};   // 6:00 AM - 10:30 AM
    m_mealTimeRanges[MealPeriod::LUNCH] = {11, 0, 14, 30};      // 11:00 AM - 2:30 PM
    m_mealTimeRanges[MealPeriod::DINNER] = {17, 0, 21, 0};      // 5:00 PM - 9:00 PM
    m_mealTimeRanges[MealPeriod::SNACK] = {21, 0, 23, 59};      // All other times are snacks

    // Set current meal period based on current time
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm* localTime = std::localtime(&timeT);

    int currentHour = localTime->tm_hour;
    int currentMinute = localTime->tm_min;

    m_currentMealPeriod = determineMealPeriod(currentHour, currentMinute);
}

MealPeriod WasteDatabase::determineMealPeriod(int hour, int minute) const {
    // Check if current time falls within any meal period
    for (const auto& mealRange : m_mealTimeRanges) {
        const TimeRange& range = mealRange.second;
        int startMinutes = range.startHour * 60 + range.startMinute;
        int endMinutes = range.endHour * 60 + range.endMinute;
        int currentMinutes = hour * 60 + minute;

        if (currentMinutes >= startMinutes && currentMinutes <= endMinutes) {
            return mealRange.first;
        }
    }

    // Default to SNACK if no other period matches
    return MealPeriod::SNACK;
}

MealPeriod WasteDatabase::determineMealPeriod(const std::string& timestamp) const {
    // Parse timestamp to get hour and minute
    std::tm tm = {};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

    if (ss.fail()) {
        // Invalid timestamp format, return unknown
        return MealPeriod::UNKNOWN;
    }

    return determineMealPeriod(tm.tm_hour, tm.tm_min);
}

void WasteDatabase::setMealPeriod(MealPeriod period) {
    m_currentMealPeriod = period;
}

MealPeriod WasteDatabase::getCurrentMealPeriod() const {
    return m_currentMealPeriod;
}

std::string WasteDatabase::getMealPeriodString() const {
    switch (m_currentMealPeriod) {
        case MealPeriod::BREAKFAST: return "Breakfast";
        case MealPeriod::LUNCH: return "Lunch";
        case MealPeriod::DINNER: return "Dinner";
        case MealPeriod::SNACK: return "Snack";
        default: return "Unknown";
    }
}

void WasteDatabase::addDetection(const Detection::FoodItem& item) {
    WasteEntry entry;
    entry.foodType = item.className;
    entry.weight = item.estimatedWeight;
    entry.timestamp = item.timestamp;
    entry.confidence = item.confidence;
    entry.mealPeriod = getMealPeriodString();

    // Add to database
    addEntry(entry);
}

void WasteDatabase::addDetections(const Detection::DetectionResult& detections) {
    for (const auto& item : detections) {
        addDetection(item);
    }
}

void WasteDatabase::addEntry(const WasteEntry& entry) {
    {
        std::lock_guard<std::mutex> lock(m_databaseMutex);
        m_entries.push_back(entry);
        m_statisticsDirty = true;
    }

    // Notify observers
    notifyDatabaseChanged();
}

std::vector<WasteEntry> WasteDatabase::getEntries(
    const std::string& foodType,
    const std::string& startDate,
    const std::string& endDate) {

    std::lock_guard<std::mutex> lock(m_databaseMutex);

    // Filter by food type if specified
    std::vector<WasteEntry> filteredEntries;
    if (!foodType.empty()) {
        for (const auto& entry : m_entries) {
            if (entry.foodType == foodType) {
                filteredEntries.push_back(entry);
            }
        }
    } else {
        filteredEntries = m_entries;
    }

    // Filter by date range if specified
    if (!startDate.empty() || !endDate.empty()) {
        filteredEntries = filterEntriesByDate(filteredEntries, startDate, endDate);
    }

    return filteredEntries;
}

std::vector<WasteEntry> WasteDatabase::filterEntriesByDate(
    const std::vector<WasteEntry>& entries,
    const std::string& startDate,
    const std::string& endDate) {

    std::vector<WasteEntry> result;

    // Parse dates
    std::tm startTm = {}, endTm = {};
    bool hasStartDate = false, hasEndDate = false;

    if (!startDate.empty()) {
        std::istringstream startSs(startDate);
        startSs >> std::get_time(&startTm, "%Y-%m-%d");
        hasStartDate = !startSs.fail();
    }

    if (!endDate.empty()) {
        std::istringstream endSs(endDate);
        endSs >> std::get_time(&endTm, "%Y-%m-%d");
        hasEndDate = !endSs.fail();

        // Set end time to end of day
        if (hasEndDate) {
            endTm.tm_hour = 23;
            endTm.tm_min = 59;
            endTm.tm_sec = 59;
        }
    }

    // Filter entries
    for (const auto& entry : entries) {
        std::tm entryTm = {};
        std::istringstream entrySs(entry.timestamp);
        entrySs >> std::get_time(&entryTm, "%Y-%m-%d %H:%M:%S");

        if (entrySs.fail()) {
            continue;
        }

        bool include = true;

        if (hasStartDate) {
            time_t entryTime = std::mktime(&entryTm);
            time_t startTime = std::mktime(&startTm);
            include = include && (entryTime >= startTime);
        }

        if (hasEndDate) {
            time_t entryTime = std::mktime(&entryTm);
            time_t endTime = std::mktime(&endTm);
            include = include && (entryTime <= endTime);
        }

        if (include) {
            result.push_back(entry);
        }
    }

    return result;
}

WasteStatistics WasteDatabase::getStatistics(TimePeriod period) {
    std::lock_guard<std::mutex> lock(m_databaseMutex);

    // If statistics are dirty or a specific period is requested, recalculate
    if (m_statisticsDirty || period != TimePeriod::ALL_TIME) {
        calculateStatistics();
        m_statisticsDirty = false;
    }

    // If ALL_TIME, return the cached statistics
    if (period == TimePeriod::ALL_TIME) {
        return m_statistics;
    }

    // Otherwise, calculate period-specific statistics
    WasteStatistics periodStats;

    // Set time period boundary
    auto now = std::chrono::system_clock::now();
    auto startTime = now;

    switch (period) {
        case TimePeriod::DAY:
            startTime = now - std::chrono::hours(24);
            break;
        case TimePeriod::WEEK:
            startTime = now - std::chrono::hours(24 * 7);
            break;
        case TimePeriod::MONTH:
            startTime = now - std::chrono::hours(24 * 30);
            break;
        case TimePeriod::YEAR:
            startTime = now - std::chrono::hours(24 * 365);
            break;
        default:
            break;
    }

    auto timeT = std::chrono::system_clock::to_time_t(startTime);
    std::tm* startTm = std::localtime(&timeT);

    std::stringstream ss;
    ss << std::put_time(startTm, "%Y-%m-%d");
    std::string startDateStr = ss.str();

    // Get entries for this period
    auto periodEntries = getEntries("", startDateStr, "");

    // Calculate statistics for this period
    periodStats.totalItems = periodEntries.size();
    periodStats.totalWeight = 0;

    for (const auto& entry : periodEntries) {
        periodStats.totalWeight += entry.weight;

        // Update weight by type
        periodStats.weightByType[entry.foodType] += entry.weight;
        periodStats.countByType[entry.foodType]++;

        // Update weight by meal
        periodStats.weightByMeal[entry.mealPeriod] += entry.weight;
    }

    // Calculate top wasted foods
    std::vector<std::pair<std::string, float>> foodWeights;
    for (const auto& item : periodStats.weightByType) {
        foodWeights.push_back({item.first, item.second});
    }

    std::sort(foodWeights.begin(), foodWeights.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

    periodStats.topWastedFoods.clear();
    for (size_t i = 0; i < std::min(size_t(5), foodWeights.size()); i++) {
        periodStats.topWastedFoods.push_back(foodWeights[i].first);
    }

    return periodStats;
}

void WasteDatabase::calculateStatistics() {
    // Reset statistics
    m_statistics = WasteStatistics();

    // Return if no entries
    if (m_entries.empty()) {
        return;
    }

    // Calculate total statistics
    m_statistics.totalItems = m_entries.size();

    // Maps for aggregating data
    std::map<std::string, float> weightByType;
    std::map<std::string, int> countByType;
    std::map<std::string, float> weightByMeal;
    std::map<std::string, float> weightByDay;
    std::map<std::string, float> weightByMonth;

    // Maps for calculating trends
    std::map<std::string, float> dailyWeight;
    std::map<std::string, float> weeklyWeight;
    std::map<std::string, float> monthlyWeight;

    // Process all entries
    for (const auto& entry : m_entries) {
        // Calculate total weight
        m_statistics.totalWeight += entry.weight;

        // Update weight by type
        weightByType[entry.foodType] += entry.weight;
        countByType[entry.foodType]++;

        // Update weight by meal period
        weightByMeal[entry.mealPeriod] += entry.weight;

        // Parse timestamp for time-based statistics
        std::tm tm = {};
        std::istringstream ss(entry.timestamp);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

        if (!ss.fail()) {
            // Get day of week
            std::stringstream dayStream;
            dayStream << std::put_time(&tm, "%A");
            std::string dayOfWeek = dayStream.str();
            weightByDay[dayOfWeek] += entry.weight;

            // Get month
            std::stringstream monthStream;
            monthStream << std::put_time(&tm, "%B");
            std::string month = monthStream.str();
            weightByMonth[month] += entry.weight;

            // Get date string for trend analysis
            std::stringstream dateStream;
            dateStream << std::put_time(&tm, "%Y-%m-%d");
            std::string dateStr = dateStream.str();
            dailyWeight[dateStr] += entry.weight;

            // Get week string (year-week)
            int weekNum = (tm.tm_yday + 7 - (tm.tm_wday == 0 ? 6 : tm.tm_wday - 1)) / 7;
            std::stringstream weekStream;
            weekStream << std::put_time(&tm, "%Y") << "-W" << std::setw(2) << std::setfill('0') << weekNum;
            std::string weekStr = weekStream.str();
            weeklyWeight[weekStr] += entry.weight;

            // Get month string (year-month)
            std::stringstream monthOnlyStream;
            monthOnlyStream << std::put_time(&tm, "%Y-%m");
            std::string monthStr = monthOnlyStream.str();
            monthlyWeight[monthStr] += entry.weight;
        }
    }

    // Store calculated statistics
    m_statistics.weightByType = weightByType;
    m_statistics.countByType = countByType;
    m_statistics.weightByMeal = weightByMeal;
    m_statistics.weightByDay = weightByDay;
    m_statistics.weightByMonth = weightByMonth;

    // Calculate top wasted foods
    std::vector<std::pair<std::string, float>> foodWeights;
    for (const auto& item : weightByType) {
        foodWeights.push_back({item.first, item.second});
    }

    std::sort(foodWeights.begin(), foodWeights.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

    m_statistics.topWastedFoods.clear();
    for (size_t i = 0; i < std::min(size_t(5), foodWeights.size()); i++) {
        m_statistics.topWastedFoods.push_back(foodWeights[i].first);
    }

    // Convert trend maps to vectors for charting
    // This ensures they're sorted by date

    // Daily trend (last 30 days)
    std::vector<std::string> dates;
    auto now = std::chrono::system_clock::now();

    for (int i = 29; i >= 0; i--) {
        auto day = now - std::chrono::hours(24 * i);
        auto timeT = std::chrono::system_clock::to_time_t(day);
        std::tm* tm = std::localtime(&timeT);

        std::stringstream ss;
        ss << std::put_time(tm, "%Y-%m-%d");
        dates.push_back(ss.str());
    }

    m_statistics.dailyTrend.clear();
    for (const auto& date : dates) {
        m_statistics.dailyTrend.push_back(dailyWeight.count(date) ? dailyWeight[date] : 0.0f);
    }

    // Calculate waste reduction (compare last week to previous week)
    float lastWeekWeight = 0.0f;
    float previousWeekWeight = 0.0f;

    auto oneWeekAgo = now - std::chrono::hours(24 * 7);
    auto twoWeeksAgo = now - std::chrono::hours(24 * 14);

    for (const auto& entry : m_entries) {
        std::tm tm = {};
        std::istringstream ss(entry.timestamp);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

        if (!ss.fail()) {
            time_t entryTime = std::mktime(&tm);
            time_t oneWeekAgoTime = std::chrono::system_clock::to_time_t(oneWeekAgo);
            time_t twoWeeksAgoTime = std::chrono::system_clock::to_time_t(twoWeeksAgo);

            if (entryTime >= oneWeekAgoTime) {
                lastWeekWeight += entry.weight;
            } else if (entryTime >= twoWeeksAgoTime) {
                previousWeekWeight += entry.weight;
            }
        }
    }

    if (previousWeekWeight > 0) {
        m_statistics.wasteSavedTotal = previousWeekWeight - lastWeekWeight;
        if (m_statistics.wasteSavedTotal > 0) {
            m_statistics.wasteSavedPercentage = (m_statistics.wasteSavedTotal / previousWeekWeight) * 100.0f;
        } else {
            m_statistics.wasteSavedTotal = 0;
            m_statistics.wasteSavedPercentage = 0;
        }
    }
}

std::vector<std::string> WasteDatabase::getTopWastedFoods(int limit) {
    std::lock_guard<std::mutex> lock(m_databaseMutex);

    // Ensure statistics are up to date
    if (m_statisticsDirty) {
        calculateStatistics();
        m_statisticsDirty = false;
    }

    // Return top foods limited to requested count
    std::vector<std::string> topFoods;
    for (size_t i = 0; i < std::min(static_cast<size_t>(limit), m_statistics.topWastedFoods.size()); i++) {
        topFoods.push_back(m_statistics.topWastedFoods[i]);
    }

    return topFoods;
}

float WasteDatabase::getTotalWasteWeight(TimePeriod period) {
    return getStatistics(period).totalWeight;
}

float WasteDatabase::getAverageWastePerDay(TimePeriod period) {
    // Calculate date range for the period
    int days = 0;
    switch (period) {
        case TimePeriod::DAY: days = 1; break;
        case TimePeriod::WEEK: days = 7; break;
        case TimePeriod::MONTH: days = 30; break;
        case TimePeriod::YEAR: days = 365; break;
        default: days = 0;
    }

    if (days == 0) {
        // For ALL_TIME, calculate actual days in the database
        std::lock_guard<std::mutex> lock(m_databaseMutex);

        if (m_entries.empty()) {
            return 0.0f;
        }

        std::string firstDate, lastDate;
        for (const auto& entry : m_entries) {
            if (firstDate.empty() || entry.timestamp < firstDate) {
                firstDate = entry.timestamp;
            }
            if (lastDate.empty() || entry.timestamp > lastDate) {
                lastDate = entry.timestamp;
            }
        }

        // Calculate days between first and last date
        std::tm firstTm = {}, lastTm = {};
        std::istringstream firstSs(firstDate), lastSs(lastDate);
        firstSs >> std::get_time(&firstTm, "%Y-%m-%d %H:%M:%S");
        lastSs >> std::get_time(&lastTm, "%Y-%m-%d %H:%M:%S");

        if (firstSs.fail() || lastSs.fail()) {
            return 0.0f;
        }

        time_t firstTime = std::mktime(&firstTm);
        time_t lastTime = std::mktime(&lastTm);

        days = static_cast<int>((lastTime - firstTime) / (60 * 60 * 24)) + 1;
        if (days <= 0) days = 1;  // Avoid division by zero
    }

    // Get total weight for the period and calculate average
    float totalWeight = getTotalWasteWeight(period);
    return totalWeight / days;
}

std::map<std::string, float> WasteDatabase::getWasteByType(TimePeriod period) {
    return getStatistics(period).weightByType;
}

std::map<std::string, float> WasteDatabase::getWasteByMeal(TimePeriod period) {
    return getStatistics(period).weightByMeal;
}

std::map<std::string, float> WasteDatabase::getWasteTrend(TimePeriod period) {
    std::map<std::string, float> trend;
    WasteStatistics stats = getStatistics(period);

    // Determine trend based on period
    if (period == TimePeriod::DAY || period == TimePeriod::WEEK) {
        // Return hourly trend for day or week
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        std::tm* today = std::localtime(&timeT);

        for (int hour = 0; hour < 24; hour++) {
            // Format hour string
            std::stringstream ss;
            ss << std::setw(2) << std::setfill('0') << hour << ":00";
            trend[ss.str()] = 0.0f;
        }

        // Get entries for today
        std::stringstream dateSs;
        dateSs << std::put_time(today, "%Y-%m-%d");
        std::string dateStr = dateSs.str();

        auto todayEntries = getEntries("", dateStr, dateStr);

        // Group by hour
        for (const auto& entry : todayEntries) {
            std::tm tm = {};
            std::istringstream ss(entry.timestamp);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

            if (!ss.fail()) {
                std::stringstream hourSs;
                hourSs << std::setw(2) << std::setfill('0') << tm.tm_hour << ":00";
                trend[hourSs.str()] += entry.weight;
            }
        }
    } else {
        // For longer periods, use the daily trend data
        auto now = std::chrono::system_clock::now();

        int days = 0;
        switch (period) {
            case TimePeriod::MONTH: days = 30; break;
            case TimePeriod::YEAR: days = 365; break;
            default: days = 7; // Default to a week
        }

        for (int i = days - 1; i >= 0; i--) {
            auto day = now - std::chrono::hours(24 * i);
            auto timeT = std::chrono::system_clock::to_time_t(day);
            std::tm* tm = std::localtime(&timeT);

            std::stringstream ss;
            ss << std::put_time(tm, "%Y-%m-%d");
            std::string dateStr = ss.str();

            // Initialize with zero
            trend[dateStr] = 0.0f;
        }

        // Fill in data from actual entries
        auto entries = getEntries();
        for (const auto& entry : entries) {
            std::tm tm = {};
            std::istringstream ss(entry.timestamp);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

            if (!ss.fail()) {
                std::stringstream dateSs;
                dateSs << std::put_time(&tm, "%Y-%m-%d");
                std::string dateStr = dateSs.str();

                if (trend.find(dateStr) != trend.end()) {
                    trend[dateStr] += entry.weight;
                }
            }
        }
    }

    return trend;
}

bool WasteDatabase::saveToFile() {
    std::lock_guard<std::mutex> lock(m_databaseMutex);

    try {
        std::ofstream file(m_databasePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open database file for writing: " << m_databasePath << std::endl;
            return false;
        }

        // Write header
        file << "FoodType,Weight,Timestamp,Confidence,MealPeriod,ImageFilename\n";

        // Write entries
        for (const auto& entry : m_entries) {
            file << entry.foodType << ","
                 << entry.weight << ","
                 << entry.timestamp << ","
                 << entry.confidence << ","
                 << entry.mealPeriod << ","
                 << entry.imageFilename << "\n";
        }

        file.close();
        std::cout << "Database saved to " << m_databasePath << " with " << m_entries.size() << " entries" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving database: " << e.what() << std::endl;
        return false;
    }
}

bool WasteDatabase::loadFromFile() {
    std::lock_guard<std::mutex> lock(m_databaseMutex);

    try {
        std::ifstream file(m_databasePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open database file for reading: " << m_databasePath << std::endl;
            return false;
        }

        m_entries.clear();

        // Read header
        std::string line;
        std::getline(file, line); // Skip header

        // Read entries
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            WasteEntry entry;

            // Parse CSV line
            if (std::getline(ss, token, ',')) entry.foodType = token;
            if (std::getline(ss, token, ',')) entry.weight = std::stof(token);
            if (std::getline(ss, token, ',')) entry.timestamp = token;
            if (std::getline(ss, token, ',')) entry.confidence = std::stof(token);
            if (std::getline(ss, token, ',')) entry.mealPeriod = token;
            if (std::getline(ss, token, ',')) entry.imageFilename = token;

            m_entries.push_back(entry);
        }

        file.close();
        m_statisticsDirty = true;

        std::cout << "Loaded " << m_entries.size() << " entries from database" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading database: " << e.what() << std::endl;
        return false;
    }
}

bool WasteDatabase::exportToCSV(const std::string& filePath) {
    std::lock_guard<std::mutex> lock(m_databaseMutex);

    try {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open export file for writing: " << filePath << std::endl;
            return false;
        }

        // Write header with additional statistical columns
        file << "FoodType,Weight,Timestamp,MealPeriod,DayOfWeek,Month\n";

        // Write entries with calculated fields
        for (const auto& entry : m_entries) {
            // Parse timestamp to extract day and month
            std::tm tm = {};
            std::istringstream ss(entry.timestamp);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

            std::string dayOfWeek = "Unknown";
            std::string month = "Unknown";

            if (!ss.fail()) {
                std::stringstream dayStream;
                dayStream << std::put_time(&tm, "%A");
                dayOfWeek = dayStream.str();

                std::stringstream monthStream;
                monthStream << std::put_time(&tm, "%B");
                month = monthStream.str();
            }

            file << entry.foodType << ","
                 << entry.weight << ","
                 << entry.timestamp << ","
                 << entry.mealPeriod << ","
                 << dayOfWeek << ","
                 << month << "\n";
        }

        file.close();
        std::cout << "Data exported to CSV: " << filePath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error exporting data: " << e.what() << std::endl;
        return false;
    }
}

bool WasteDatabase::exportToJSON(const std::string& filePath) {
    std::lock_guard<std::mutex> lock(m_databaseMutex);

    try {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file for writing: " << filePath << std::endl;
            return false;
        }

        // Ensure statistics are up to date
        if (m_statisticsDirty) {
            calculateStatistics();
            m_statisticsDirty = false;
        }

        // Write JSON start
        file << "{\n";

        // Write metadata
        file << "  \"metadata\": {\n";
        file << "    \"totalEntries\": " << m_entries.size() << ",\n";
        file << "    \"totalWeight\": " << m_statistics.totalWeight << ",\n";
        file << "    \"exportDate\": \"" << getCurrentTimestamp() << "\"\n";
        file << "  },\n";

        // Write entries
        file << "  \"entries\": [\n";
        for (size_t i = 0; i < m_entries.size(); i++) {
            const auto& entry = m_entries[i];

            file << "    {\n";
            file << "      \"foodType\": \"" << entry.foodType << "\",\n";
            file << "      \"weight\": " << entry.weight << ",\n";
            file << "      \"timestamp\": \"" << entry.timestamp << "\",\n";
            file << "      \"confidence\": " << entry.confidence << ",\n";
            file << "      \"mealPeriod\": \"" << entry.mealPeriod << "\"";

            if (!entry.imageFilename.empty()) {
                file << ",\n      \"imageFilename\": \"" << entry.imageFilename << "\"";
            }

            file << "\n    }";

            if (i < m_entries.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        file << "  ],\n";

        // Write statistics
        file << "  \"statistics\": {\n";
        file << "    \"topWastedFoods\": [\n";
        for (size_t i = 0; i < m_statistics.topWastedFoods.size(); i++) {
            file << "      \"" << m_statistics.topWastedFoods[i] << "\"";
            if (i < m_statistics.topWastedFoods.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        file << "    ],\n";

        // Write waste by type
        file << "    \"wasteByType\": {\n";
        size_t i = 0;
        for (const auto& [type, weight] : m_statistics.weightByType) {
            file << "      \"" << type << "\": " << weight;
            if (i < m_statistics.weightByType.size() - 1) {
                file << ",";
            }
            file << "\n";
            i++;
        }
        file << "    },\n";

        // Write waste by meal
        file << "    \"wasteByMeal\": {\n";
        i = 0;
        for (const auto& [meal, weight] : m_statistics.weightByMeal) {
            file << "      \"" << meal << "\": " << weight;
            if (i < m_statistics.weightByMeal.size() - 1) {
                file << ",";
            }
            file << "\n";
            i++;
        }
        file << "    },\n";

        // Write waste by day
        file << "    \"wasteByDay\": {\n";
        i = 0;
        for (const auto& [day, weight] : m_statistics.weightByDay) {
            file << "      \"" << day << "\": " << weight;
            if (i < m_statistics.weightByDay.size() - 1) {
                file << ",";
            }
            file << "\n";
            i++;
        }
        file << "    },\n";

        // Write waste reduction
        file << "    \"wasteReduction\": {\n";
        file << "      \"savedTotal\": " << m_statistics.wasteSavedTotal << ",\n";
        file << "      \"savedPercentage\": " << m_statistics.wasteSavedPercentage << "\n";
        file << "    }\n";
        file << "  }\n";

        // Close JSON
        file << "}\n";

        file.close();
        std::cout << "Data exported to JSON: " << filePath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error exporting to JSON: " << e.what() << std::endl;
        return false;
    }
}

void WasteDatabase::saveDetectionImage(const cv::Mat& frame, const Detection::FoodItem& item, std::string& outputPath) {
    // Create images directory if it doesn't exist
    fs::path dbPath(m_databasePath);
    fs::path imagesDir = dbPath.parent_path() / "images";

    try {
        if (!fs::exists(imagesDir)) {
            fs::create_directories(imagesDir);
        }

        // Generate filename based on timestamp and food type
        std::string timestamp = item.timestamp;
        std::replace(timestamp.begin(), timestamp.end(), ' ', '_');
        std::replace(timestamp.begin(), timestamp.end(), ':', '-');

        std::stringstream ss;
        ss << "food_waste_" << item.className << "_" << timestamp << ".jpg";
        std::string filename = ss.str();

        fs::path imagePath = imagesDir / filename;

        // Extract the region of interest
        cv::Rect box = item.boundingBox;
        // Ensure the box is within the frame boundaries
        box.x = std::max(0, box.x);
        box.y = std::max(0, box.y);
        box.width = std::min(box.width, frame.cols - box.x);
        box.height = std::min(box.height, frame.rows - box.y);

        if (box.width <= 0 || box.height <= 0) {
            std::cerr << "Invalid bounding box for image saving" << std::endl;
            outputPath = "";
            return;
        }

        // Extract the region and save it
        cv::Mat roi = frame(box);
        cv::imwrite(imagePath.string(), roi);

        outputPath = imagePath.string();
        std::cout << "Saved detection image to " << outputPath << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving detection image: " << e.what() << std::endl;
        outputPath = "";
    }
}

void WasteDatabase::registerChangeCallback(DatabaseChangeCallback callback) {
    m_changeCallbacks.push_back(callback);
}

void WasteDatabase::notifyDatabaseChanged() {
    for (const auto& callback : m_changeCallbacks) {
        callback();
    }
}

std::string WasteDatabase::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}