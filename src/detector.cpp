#include "include/cnn_detect/detector.h"

const int resize_to=300;
const float threshold=0.5;
const int cpu_threads=16;
const char* model_path = "/home/yamabuki/Downloads/cnn_files/super_resolution.onnx";
//const char* img_path="/home/yamabuki/Downloads/cnn_files/IMG_20181228_102706.jpg";



Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
Ort::SessionOptions session_options;
Ort::Session session(env, model_path, session_options);
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
std::vector<int64_t> input_dims;
std::vector<const char*> input_node_names;
std::vector<const char*> output_node_names;


void imgProcess(cv::Mat& img)
{
    cv::resize(img,img,cv::Size(resize_to,resize_to));
    img/=255;
}

void receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
    cv_bridge::CvImagePtr cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
    cv::Mat img=cv_image_->image.clone();
    imgProcess(img);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1, cv::Size(resize_to, resize_to));

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

    auto* data_dim1 = output_tensors[0].GetTensorMutableData<float>();
    auto* data_dim2 = output_tensors[1].GetTensorMutableData<int>();
    auto* data_dim3 = output_tensors[2].GetTensorMutableData<float>();

    std::vector<int> index_vec;

    std::vector<double> prob_vec;
    for (int i = 0; i < 5; i++)
    {
        if(data_dim3[i]>=threshold)
        {
            prob_vec.push_back(data_dim3[i]);
            index_vec.push_back(i);
        }
    }
    if (prob_vec.size()!=0)
    {
        int target_num=index_vec.size();

        std::vector<double> box_vec;
        for (int i = 0; i <target_num; i++)
        {
            box_vec.push_back(data_dim1[index_vec[i]]*cv_image_->image.cols/resize_to);
            box_vec.push_back(data_dim1[index_vec[i]+1]*cv_image_->image.rows/resize_to);
            box_vec.push_back(data_dim1[index_vec[i]+2]*cv_image_->image.cols/resize_to);
            box_vec.push_back(data_dim1[index_vec[i]+3]*cv_image_->image.rows/resize_to);
        }

        std::vector<int> label_vec;
        for (int i = 0; i <target_num; i++)
        {
            label_vec.push_back(data_dim2[index_vec[i]]);
        }

        for (int i=0;i<target_num*4;i+=4)
        {
            cv::rectangle(cv_image_->image,cv::Point2d(box_vec[i],box_vec[i+1]),cv::Point2d(box_vec[i+2],box_vec[i+3]),cv::Scalar(255,0,0),2);
        }
        img_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
    }
    else
    {
        img_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
    }
}


int main(int argc, char ** argv)
{

    session_options.SetIntraOpNumThreads(cpu_threads);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();

    size_t num_output_nodes = session.GetOutputCount();
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of output = %zu\n", num_output_nodes);

    const char* input_name = session.GetInputName(0, allocator);

    const char* output_name = session.GetOutputName(0, allocator);
    const char* output_name1 = session.GetOutputName(1, allocator);
    const char* output_name2= session.GetOutputName(2, allocator);

    input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_node_names = { input_name};
    output_node_names = { output_name,output_name1,output_name2};

    ros::init(argc, argv,"cnn_node");
    ros::NodeHandle nh;
    ros::Subscriber img_subscriber=nh.subscribe<sensor_msgs::Image>("/hk_camera/image_raw", 1, &receiveFromCam);
    img_publisher=nh.advertise<sensor_msgs::Image>("cnn_publisher",1);

    while (ros::ok())
    {
        ros::spinOnce();
    }
    return 0;
}