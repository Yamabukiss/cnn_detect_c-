#include "include/cnn_detect/detector.h"

int resize_to=300;

void img_process(cv::Mat& img)
{
    cv::resize(img,img,cv::Size(resize_to,resize_to));
    img/=255;
}

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(16);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const char* model_path = "/home/yamabuki/Downloads/cnn_files/super_resolution.onnx";

    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();

    size_t num_output_nodes = session.GetOutputCount();
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of output = %zu\n", num_output_nodes);

    const char* input_name = session.GetInputName(0, allocator);

    // when output is 3 dims
    const char* output_name = session.GetOutputName(0, allocator);
    const char* output_name1 = session.GetOutputName(1, allocator);
    const char* output_name2= session.GetOutputName(2, allocator);

    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<const char*> input_node_names = { input_name};
    std::vector<const char*> output_node_names = { output_name,output_name1,output_name2};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    cv::Mat origin_img = cv::imread("/home/yamabuki/Downloads/cnn_files/IMG_20181228_102706.jpg");
    cv::Mat img=origin_img.clone();
    img_process(img);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1, cv::Size(resize_to, resize_to));

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
    
    auto* data_dim1 = output_tensors[0].GetTensorMutableData<float>();
    auto* data_dim2 = output_tensors[1].GetTensorMutableData<int>();
    auto* data_dim3 = output_tensors[2].GetTensorMutableData<float>();

    cv::Mat box_mat = cv::Mat_<int>(1, 20);
    for (int i = 0; i < box_mat.rows; i++)
    {
        for (int j = 0; j < box_mat.cols; j++)
        {
            if(j%2==0) box_mat.at<int>(i, j) = data_dim1[j]*origin_img.cols/resize_to;
            else box_mat.at<int>(i, j) = data_dim1[j]*origin_img.rows/resize_to;
        }
    }

    cv::Mat label_mat = cv::Mat_<int>(1, 5);
    for (int i = 0; i < label_mat.rows; i++)
    {
        for (int j = 0; j < label_mat.cols; j++)
        {
            label_mat.at<int>(i, j) = data_dim2[j];
        }
    }

    cv::Mat prob_mat = cv::Mat_<double>(1, 5);
    for (int i = 0; i < prob_mat.rows; i++)
    {
        for (int j = 0; j < prob_mat.cols; j++)
        {
            prob_mat.at<double>(i, j) = data_dim3[j];
        }
    }

    std::cout<<box_mat<<std::endl;
    std::cout<<label_mat<<std::endl;
    std::cout<<prob_mat<<std::endl;

    cv::rectangle(origin_img,cv::Point2d(box_mat.at<int>(0,0),box_mat.at<int>(0,1)),cv::Point2d(box_mat.at<int>(0,2),box_mat.at<int>(0,3)),cv::Scalar(255,0,0),2);
    cv::imshow("output",origin_img);
    cv::waitKey(0);
    return 0;
}
