#include "include/cnn_detect/detector.h"

namespace cnn_detect
{
    Detector::Detector() 
    {
        model_path_ = "/home/yamabuki/Downloads/micro.onnx";
        cpu_threads_=16;
        threshold_=0.5;
        resize_to_=200;
        memoryInfo_pointer_= nullptr;
        session_pointer_= nullptr;
    }

    void Detector::onInit()
    {
        ros::NodeHandle nh = getMTPrivateNodeHandle();
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
        Ort::SessionOptions session_options;
        
        session_options.SetIntraOpNumThreads(cpu_threads_);

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        static Ort::Session session(env, cnn_detect::Detector::model_path_, session_options);
        session_pointer_=&session;
        static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        memoryInfo_pointer_=&memory_info;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();

        size_t num_output_nodes = session.GetOutputCount();
        printf("Number of inputs = %zu\n", num_input_nodes);
        printf("Number of output = %zu\n", num_output_nodes);

        const char *input_name = session.GetInputName(0, allocator);

        const char *output_name = session.GetOutputName(0, allocator);
        const char *output_name1 = session.GetOutputName(1, allocator);
        const char *output_name2 = session.GetOutputName(2, allocator);

        input_dims_ = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_node_names_ = {input_name};
        output_node_names_ = {output_name, output_name1, output_name2};

        img_subscriber_ = nh.subscribe<sensor_msgs::Image>("/hk_camera/image_raw", 1, &Detector::receiveFromCam,this);
        img_publisher_ = nh.advertise<sensor_msgs::Image>("cnn_publisher", 1);

    }
    
    void Detector::imgProcess(cv::Mat &img) {
        cv::resize(img, img, cv::Size(resize_to_, resize_to_));
        img /= 255;
    }

    void Detector::receiveFromCam(const sensor_msgs::ImageConstPtr &image) {
        cv_bridge::CvImagePtr cv_image_ = boost::make_shared<cv_bridge::CvImage>(
                *cv_bridge::toCvShare(image, image->encoding));
        cv::Mat img = cv_image_->image.clone();
        imgProcess(img);
        cv::Mat blob = cv::dnn::blobFromImage(img, 1, cv::Size(resize_to_, resize_to_));

        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(
                Ort::Value::CreateTensor<float>(*memoryInfo_pointer_, blob.ptr<float>(), blob.total(), input_dims_.data(),
                                                input_dims_.size()));

        auto output_tensors = session_pointer_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), input_tensors.data(),
                                          input_node_names_.size(), output_node_names_.data(), output_node_names_.size());
        std::cout << output_tensors[0] << std::endl;
        std::cout << output_tensors[1] << std::endl;
        std::cout << output_tensors[2] << std::endl;
        auto *data_dim1 = output_tensors[0].GetTensorMutableData<float>();
        auto *data_dim2 = output_tensors[1].GetTensorMutableData<float>();
        auto *data_dim3 = output_tensors[2].GetTensorMutableData<int>();

        std::vector<int> index_vec;

        std::vector<double> prob_vec;
        for (int i = 0; i < 5; i++) {
            if (data_dim3[i] >= threshold_) {
                prob_vec.push_back(data_dim3[i]);
                index_vec.push_back(i);
            }
        }
        if (prob_vec.size() != 0) {
            int target_num = index_vec.size();

            std::vector<double> box_vec;
            for (int i = 0; i < target_num; i++) {
                box_vec.push_back(data_dim1[index_vec[i]] * cv_image_->image.cols / resize_to_);
                box_vec.push_back(data_dim1[index_vec[i] + 1] * cv_image_->image.rows / resize_to_);
                box_vec.push_back(data_dim1[index_vec[i] + 2] * cv_image_->image.cols / resize_to_);
                box_vec.push_back(data_dim1[index_vec[i] + 3] * cv_image_->image.rows / resize_to_);
            }

            std::vector<int> label_vec;
            for (int i = 0; i < target_num; i++) {
                label_vec.push_back(data_dim2[index_vec[i]]);
            }

            for (int i = 0; i < target_num * 4; i += 4) {
                cv::rectangle(cv_image_->image, cv::Point2d(box_vec[i], box_vec[i + 1]),
                              cv::Point2d(box_vec[i + 2], box_vec[i + 3]), cv::Scalar(255, 0, 0), 2);
            }
            img_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
        } else {
            img_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
        }
    }
    Detector::~Detector()
    {
    }
}
PLUGINLIB_EXPORT_CLASS(cnn_detect::Detector, nodelet::Nodelet)