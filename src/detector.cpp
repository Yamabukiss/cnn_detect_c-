#include "include/cnn_detect/detector.h"
using namespace cv;

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

    const char* output_name = session.GetOutputName(0, allocator);
    const char* output_name1 = session.GetOutputName(1, allocator);
    const char* output_name2= session.GetOutputName(2, allocator);

    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<const char*> input_node_names = { input_name};
    std::vector<const char*> output_node_names = { output_name,output_name1,output_name2};


    Mat img = imread("/home/yamabuki/Downloads/cnn_files/IMG_20181228_102706.jpg");
    Mat re_img;
    cv::resize(img,re_img,cv::Size(300,300));
    Mat det1;
    cv::resize(img, det1, Size(300, 300), INTER_AREA);
    det1/=255;
    Mat blob = dnn::blobFromImage(det1, 1, Size(300, 300));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
    assert(output_tensors.size() == 3 && output_tensors.front().IsTensor());
    auto* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto* floatarr1 = output_tensors[1].GetTensorMutableData<float>();
    auto* floatarr2 = output_tensors[2].GetTensorMutableData<float>();
    Mat newarr = Mat_<int>(1, 20); //定义一个1*1000的矩阵
    for (int i = 0; i < newarr.rows; i++)
    {
        for (int j = 0; j < newarr.cols; j++) //矩阵列数循环
        {
            newarr.at<int>(i, j) = floatarr[j];
        }
            }

    Mat newarr1 = Mat_<int>(1, 5); //定义一个1*1000的矩阵
    for (int i = 0; i < newarr1.rows; i++)
    {
        for (int j = 0; j < newarr1.cols; j++) //矩阵列数循环
        {
            newarr1.at<int>(i, j) = floatarr1[j];
        }
    }

    Mat newarr2 = Mat_<double>(1, 5); //定义一个1*1000的矩阵
    for (int i = 0; i < newarr2.rows; i++)
    {
        for (int j = 0; j < newarr2.cols; j++) //矩阵列数循环
        {
            newarr2.at<double>(i, j) = floatarr2[j];
        }
    }

    std::cout<<newarr<<std::endl;
    std::cout<<newarr1<<std::endl;
    std::cout<<newarr2<<std::endl;

    cv::rectangle(re_img,cv::Point2d(newarr.at<int>(0,0),newarr.at<int>(0,1)),cv::Point2d(newarr.at<int>(0,2),newarr.at<int>(0,3)),cv::Scalar(255,0,0),2);
    cv::imshow("output",re_img);
    cv::waitKey(0);
    return 0;
}
