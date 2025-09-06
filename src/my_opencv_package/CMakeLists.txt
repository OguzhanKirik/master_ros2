cmake_minimum_required(VERSION 3.8)
project(my_opencv_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)

# Add executable
add_executable(image_processor src/image_processing.cpp)

# Target dependencies
ament_target_dependencies(image_processor 
  rclcpp 
  sensor_msgs 
  cv_bridge 
  OpenCV
  std_msgs
)

# Install targets
install(TARGETS
  image_processor
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
