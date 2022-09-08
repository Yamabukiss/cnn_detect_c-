cmake_minimum_required(VERSION 3.0.2)
project(cnn_detect)

#******onnxruntime*****
set(ONNXRUNTIME_ROOT_PATH /home/yamabuki/Downloads/onnxruntime-linux-x64-1.11.1)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)


find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
        )
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)


catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        CATKIN_DEPENDS
        roscpp
        dynamic_reconfigure
        std_msgs
        cv_bridge
        LIBRARIES ${PROJECT_NAME}
)
#
#include_directories(
#        include
#        ${catkin_INCLUDE_DIRS}
#)
## Declare a C++ library
add_library(${PROJECT_NAME} src/detector.cpp )
#add_executable(${PROJECT_NAME} src/test_node.cpp src/detector.cpp)



add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        )