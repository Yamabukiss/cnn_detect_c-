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



void img_process(cv::Mat& img);


//namespace cnn_detector
//{
//    class Detector : public nodelet::Nodelet
//    {
//    public:
//        Detector();
//        virtual ~Detector();
//        void onInit() override;
//        void receiveFromCam(const sensor_msgs::ImageConstPtr& image);
//    };
//}