/**
 * Configuration Loader Header
 *
 * Handles loading and saving application configuration from JSON files
 */

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <map>
#include <vector>

namespace Utils {

class ConfigLoader {
public:
    explicit ConfigLoader(const std::string& configPath);

    // Load and save config
    bool loadConfig();
    bool saveConfig() const;

    // Camera settings
    int getCameraIndex() const;
    void setCameraIndex(int index);

    // Paths
    std::string getDatabasePath() const;
    void setDatabasePath(const std::string& path);

    std::string getModelPath() const;
    void setModelPath(const std::string& path);

    std::string getClassesPath() const;
    void setClassesPath(const std::string& path);

    std::string getTrainingDataPath() const;
    void setTrainingDataPath(const std::string& path);

    // Detection settings
    float getConfidenceThreshold() const;
    void setConfidenceThreshold(float threshold);

    // Training settings
    float getLearningRate() const;
    void setLearningRate(float rate);

    int getTrainingIntervalHours() const;
    void setTrainingIntervalHours(int hours);

    // UI settings
    bool getShowDetectionBoxes() const;
    void setShowDetectionBoxes(bool show);

    bool getShowStatistics() const;
    void setShowStatistics(bool show);

    // Generic property access
    template<typename T>
    T getValue(const std::string& key, const T& defaultValue) const;

    template<typename T>
    void setValue(const std::string& key, const T& value);

private:
    // Creates default configuration
    void createDefaultConfig();

    // Config file path
    std::string m_configPath;

    // Configuration maps
    std::map<std::string, std::string> m_stringConfig;
    std::map<std::string, int> m_intConfig;
    std::map<std::string, float> m_floatConfig;
    std::map<std::string, bool> m_boolConfig;

    // Default values
    static const std::map<std::string, std::string> DEFAULT_STRING_CONFIG;
    static const std::map<std::string, int> DEFAULT_INT_CONFIG;
    static const std::map<std::string, float> DEFAULT_FLOAT_CONFIG;
    static const std::map<std::string, bool> DEFAULT_BOOL_CONFIG;
};

} // namespace Utils

#endif // CONFIG_LOADER_H