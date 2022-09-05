#include "include/cnn_detect/detector.h"
using namespace cv;
void PreProcess(const Mat& image, Mat& image_blob)
{
    Mat input;
    image.copyTo(input);


    //数据处理 标准化
    std::vector<Mat> channels, channel_p;
    split(input, channels);
    Mat R, G, B;
    B = channels.at(0);
    G = channels.at(1);
    R = channels.at(2);

    B = B / 255;
    G = G / 255;
    R = R / 255;

    channel_p.push_back(R);
    channel_p.push_back(G);
    channel_p.push_back(B);

    Mat outt;
    merge(channel_p, outt);
    image_blob = outt;
}



int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(12);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, "/home/yamabuki/Downloads/cnn_files/super_resolution.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);


    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of output = %zu\n", num_output_nodes);

    std::vector<const char*> input_node_names = { "images" };
    std::vector<const char*> output_node_names = { "3513","3481","3480"};

    //获取输入name
    const char* input_name = session.GetInputName(0, allocator);
    std::cout << "input_name:" << input_name << std::endl;
    //获取输出name
    const char* output_name = session.GetOutputName(0, allocator);
    const char* output_name2 = session.GetOutputName(1, allocator);
    const char* output_name3 = session.GetOutputName(2, allocator);
    std::cout << "output_name: " << output_name << std::endl;
    std::cout << "output_name: " << output_name2 << std::endl;
    std::cout << "output_name: " << output_name3 << std::endl;
    std::vector<const char*> input_names{ input_name };
    std::vector<const char*> output_names = { output_name };
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto i=output_dims.begin();i!=output_dims.end();i++)
    {
        std::cout<<*i<<std::endl;
    }

    Mat img = imread("/home/yamabuki/Downloads/cnn_files/IMG_20181228_102706.jpg");
    Mat det1, det2;
    resize(img, det1, Size(300, 300), INTER_AREA);
    det1.convertTo(det1, CV_32FC3);
    PreProcess(det1, det2);         //标准化处理
    Mat blob = dnn::blobFromImage(det2, 1., Size(300, 300), Scalar(0, 0, 0), false, true);
    printf("Load success!\n");

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_names.size(), output_node_names.data(), output_node_names.size());
    assert(output_tensors.size() == 3 && output_tensors.front().IsTensor());
    auto* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto* floatarr1 = output_tensors[1].GetTensorMutableData<float>();
    auto* floatarr2 = output_tensors[2].GetTensorMutableData<float>();
    std::cout<<*floatarr<<std::endl;
    std::cout<<*floatarr1<<std::endl;
    std::cout<<*floatarr2<<std::endl;

    return 0;
}