#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "sensor_msgs/CameraInfo.h"
#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace cnn_detect
{
    class Detector : public nodelet::Nodelet
    {
    public:
        Detector();
        virtual ~Detector();
        void onInit() override;
        void receiveFromCam(const sensor_msgs::ImageConstPtr& image);
        void imgProcess(cv::Mat& img);
        ros::Subscriber img_subscriber_;
        ros::Publisher img_publisher_;
        int resize_to_;
        float threshold_;
        int cpu_threads_;
        const char* model_path_;
        std::vector<int64_t> input_dims_;
        std::vector<const char *> input_node_names_;
        std::vector<const char *> output_node_names_;
        Ort::Session *session_pointer_{};
        Ort::MemoryInfo *memoryInfo_pointer_{};
    };
}
