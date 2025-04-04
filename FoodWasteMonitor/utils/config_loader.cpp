/**
 * Configuration Loader Implementation
 */

#include "config_loader.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace Utils {

// Define default configuration values
const std::map<std::string, std::string> ConfigLoader::DEFAULT_STRING_CONFIG = {
    {"database_path", "data/waste_database.csv"},
    {"model_path", "models/food_detection_model.weights"},
    {"classes_path", "models/food_classes.txt"},
    {"training_data_path", "data/training"}
};

const std::map<std::string, int> ConfigLoader::DEFAULT_INT_CONFIG = {
    {"camera_index", 0},
    {"training_interval_hours", 48}
};

const std::map<std::string, float> ConfigLoader::DEFAULT_FLOAT_CONFIG = {
    {"confidence_threshold", 0.5f},
    {"learning_rate", 0.001f}
};

const std::map<std::string, bool> ConfigLoader::DEFAULT_BOOL_CONFIG = {
    {"show_detection_boxes", true},
    {"show_statistics", true}
};

ConfigLoader::ConfigLoader(const std::string& configPath)
    : m_configPath(configPath) {

    // Try to load existing config or create default
    if (!loadConfig()) {
        createDefaultConfig();
        saveConfig();
    }
}

bool ConfigLoader::loadConfig() {
    try {
        if (!fs::exists(m_configPath)) {
            std::cout << "Config file not found: " << m_configPath << std::endl;
            return false;
        }

        std::ifstream file(m_configPath);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << m_configPath << std::endl;
            return false;
        }

        json config;
        file >> config;
        file.close();

        // Load string configuration
        for (const auto& [key, defaultValue] : DEFAULT_STRING_CONFIG) {
            m_stringConfig[key] = config.value(key, defaultValue);
        }

        // Load int configuration
        for (const auto& [key, defaultValue] : DEFAULT_INT_CONFIG) {
            m_intConfig[key] = config.value(key, defaultValue);
        }

        // Load float configuration
        for (const auto& [key, defaultValue] : DEFAULT_FLOAT_CONFIG) {
            m_floatConfig[key] = config.value(key, defaultValue);
        }

        // Load bool configuration
        for (const auto& [key, defaultValue] : DEFAULT_BOOL_CONFIG) {
            m_boolConfig[key] = config.value(key, defaultValue);
        }

        std::cout << "Loaded configuration from " << m_configPath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigLoader::saveConfig() const {
    try {
        // Create parent directory if it doesn't exist
        fs::path configPath(m_configPath);
        fs::path parentDir = configPath.parent_path();

        if (!parentDir.empty() && !fs::exists(parentDir)) {
            fs::create_directories(parentDir);
        }

        // Create JSON object
        json config;

        // Add string configuration
        for (const auto& [key, value] : m_stringConfig) {
            config[key] = value;
        }

        // Add int configuration
        for (const auto& [key, value] : m_intConfig) {
            config[key] = value;
        }

        // Add float configuration
        for (const auto& [key, value] : m_floatConfig) {
            config[key] = value;
        }

        // Add bool configuration
        for (const auto& [key, value] : m_boolConfig) {
            config[key] = value;
        }

        // Write to file
        std::ofstream file(m_configPath);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file for writing: " << m_configPath << std::endl;
            return false;
        }

        file << config.dump(4); // Pretty print with 4-space indentation
        file.close();

        std::cout << "Saved configuration to " << m_configPath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving config: " << e.what() << std::endl;
        return false;
    }
}

void ConfigLoader::createDefaultConfig() {
    // Initialize with default values
    m_stringConfig = DEFAULT_STRING_CONFIG;
    m_intConfig = DEFAULT_INT_CONFIG;
    m_floatConfig = DEFAULT_FLOAT_CONFIG;
    m_boolConfig = DEFAULT_BOOL_CONFIG;

    std::cout << "Created default configuration" << std::endl;
}

int ConfigLoader::getCameraIndex() const {
    return m_intConfig.at("camera_index");
}

void ConfigLoader::setCameraIndex(int index) {
    m_intConfig["camera_index"] = index;
}

std::string ConfigLoader::getDatabasePath() const {
    return m_stringConfig.at("database_path");
}

void ConfigLoader::setDatabasePath(const std::string& path) {
    m_stringConfig["database_path"] = path;
}

std::string ConfigLoader::getModelPath() const {
    return m_stringConfig.at("model_path");
}

void ConfigLoader::setModelPath(const std::string& path) {
    m_stringConfig["model_path"] = path;
}

std::string ConfigLoader::getClassesPath() const {
    return m_stringConfig.at("classes_path");
}

void ConfigLoader::setClassesPath(const std::string& path) {
    m_stringConfig["classes_path"] = path;
}

std::string ConfigLoader::getTrainingDataPath() const {
    return m_stringConfig.at("training_data_path");
}

void ConfigLoader::setTrainingDataPath(const std::string& path) {
    m_stringConfig["training_data_path"] = path;
}

float ConfigLoader::getConfidenceThreshold() const {
    return m_floatConfig.at("confidence_threshold");
}

void ConfigLoader::setConfidenceThreshold(float threshold) {
    m_floatConfig["confidence_threshold"] = threshold;
}

float ConfigLoader::getLearningRate() const {
    return m_floatConfig.at("learning_rate");
}

void ConfigLoader::setLearningRate(float rate) {
    m_floatConfig["learning_rate"] = rate;
}

int ConfigLoader::getTrainingIntervalHours() const {
    return m_intConfig.at("training_interval_hours");
}

void ConfigLoader::setTrainingIntervalHours(int hours) {
    m_intConfig["training_interval_hours"] = hours;
}

bool ConfigLoader::getShowDetectionBoxes() const {
    return m_boolConfig.at("show_detection_boxes");
}

void ConfigLoader::setShowDetectionBoxes(bool show) {
    m_boolConfig["show_detection_boxes"] = show;
}

bool ConfigLoader::getShowStatistics() const {
    return m_boolConfig.at("show_statistics");
}

void ConfigLoader::setShowStatistics(bool show) {
    m_boolConfig["show_statistics"] = show;
}

// Template specializations for different types
template<>
std::string ConfigLoader::getValue<std::string>(const std::string& key, const std::string& defaultValue) const {
    auto it = m_stringConfig.find(key);
    return (it != m_stringConfig.end()) ? it->second : defaultValue;
}

template<>
int ConfigLoader::getValue<int>(const std::string& key, const int& defaultValue) const {
    auto it = m_intConfig.find(key);
    return (it != m_intConfig.end()) ? it->second : defaultValue;
}

template<>
float ConfigLoader::getValue<float>(const std::string& key, const float& defaultValue) const {
    auto it = m_floatConfig.find(key);
    return (it != m_floatConfig.end()) ? it->second : defaultValue;
}

template<>
bool ConfigLoader::getValue<bool>(const std::string& key, const bool& defaultValue) const {
    auto it = m_boolConfig.find(key);
    return (it != m_boolConfig.end()) ? it->second : defaultValue;
}

template<>
void ConfigLoader::setValue<std::string>(const std::string& key, const std::string& value) {
    m_stringConfig[key] = value;
}

template<>
void ConfigLoader::setValue<int>(const std::string& key, const int& value) {
    m_intConfig[key] = value;
}

template<>
void ConfigLoader::setValue<float>(const std::string& key, const float& value) {
    m_floatConfig[key] = value;
}

template<>
void ConfigLoader::setValue<bool>(const std::string& key, const bool& value) {
    m_boolConfig[key] = value;
}

} // namespace Utils