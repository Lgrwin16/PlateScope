cmake_minimum_required(VERSION 3.10)
project(FoodWasteMonitor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(nlohmann_json QUIET)
find_package(CURL REQUIRED)

# If nlohmann_json is not found, we'll use a fetched version
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.11.2)
    FetchContent_MakeAvailable(json)
endif()

# Include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CURL_INCLUDE_DIRS}
)

# Source files
set(SOURCES
        main.cpp
        camera/camera_manager.cpp
        detection/food_detector.cpp
        data/waste_database.cpp
        analysis/stats_analyzer.cpp
        training/model_trainer.cpp
        ui/user_interface.cpp
        utils/config_loader.cpp
)

# Headers
set(HEADERS
        camera/camera_manager.h
        detection/food_detector.h
        data/waste_database.h
        analysis/stats_analyzer.h
        training/model_trainer.h
        ui/user_interface.h
        utils/config_loader.h
)

# Create executable
add_executable(food_waste_monitor ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(food_waste_monitor
        ${OpenCV_LIBS}
        ${CURL_LIBRARIES}
        nlohmann_json::nlohmann_json
)

# If on Linux, also link pthread
if(UNIX AND NOT APPLE)
    target_link_libraries(food_waste_monitor pthread)
endif()

# Install executable
install(TARGETS food_waste_monitor DESTINATION bin)

# Create directory structure
set(DIRECTORIES
        data
        models
        data/training
        data/training/images
        data/training/annotations
        data/training/checkpoints
)

# Create directories in build directory
foreach(dir ${DIRECTORIES})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${dir})
endforeach()

# Copy necessary files (if any exist)
file(GLOB MODEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/*)
if(MODEL_FILES)
    file(COPY ${MODEL_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/models)
endif()

# Create default food classes file if it doesn't exist
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/models/food_classes.txt)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/models/food_classes.txt
            "apple
banana
bread
burger
cake
carrot
chicken
cookie
fries
pasta
pizza
rice
salad
sandwich
vegetable"
    )
endif()

# Output information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
