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


// 读取txt文件


int main()         // 返回值为整型带参的main函数. 函数体内使用或不使用argc和argv都可
{

    //environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
    Ort::SessionOptions session_options;
    // 使用1个线程执行op,若想提升速度，增加线程数
    session_options.SetIntraOpNumThreads(1);
    //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    // ORT_ENABLE_ALL: 启用所有可能的优化
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //load  model and creat session

#ifdef _WIN32
    const wchar_t* model_path = L"F:\\Pycharm\\PyCharm_Study\\Others\\c++_learning\\C++_Master\\Onnx\\classification\\vgg16.onnx";
#else
    const char* model_path = "/home/yamabuki/Downloads/cnn_files/super_resolution.onnx";
#endif

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;


    //model info
    // 获得模型又多少个输入和输出，一般是指对应网络层的数目
    // 一般输入只有图像的话input_nodes为1
    size_t num_input_nodes = session.GetInputCount();
    // 如果是多输出网络，就会是对应输出的数目
    size_t num_output_nodes = session.GetOutputCount();
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of output = %zu\n", num_output_nodes);
    //获取输入name
    const char* input_name = session.GetInputName(0, allocator);
    std::cout << "input_name:" << input_name << std::endl;
    //获取输出name
    const char* output_name = session.GetOutputName(0, allocator);
    const char* output_name1 = session.GetOutputName(1, allocator);
    const char* output_name2= session.GetOutputName(2, allocator);
    std::cout << "output_name: " << output_name << std::endl;
    std::cout << "output_name: " << output_name1 << std::endl;
    std::cout << "output_name: " << output_name2 << std::endl;
    // 自动获取维度数量
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "input_dims:" << input_dims[0] << std::endl;
    std::vector<const char*> input_names{ input_name };
    std::vector<const char*> output_names = { output_name };
    std::vector<const char*> input_node_names = { "images" };
    std::vector<const char*> output_node_names = { output_name,output_name1,output_name2};


    //加载图片
    Mat img = imread("/home/yamabuki/Downloads/cnn_files/IMG_20181228_102706.jpg");
    Mat re_img;
    cv::resize(img,re_img,cv::Size(300,300));
    Mat det1, det2;
    resize(img, det1, Size(300, 300), INTER_AREA);
    det1.convertTo(det1, CV_32FC3);
    PreProcess(det1, det2);         //标准化处理
    Mat blob = dnn::blobFromImage(det2, 1., Size(300, 300), Scalar(0, 0, 0), false, true);
    printf("Load success!\n");

    //创建输入tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
    /*cout << int(input_dims.size()) << endl;*/

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_names.size(), output_node_names.data(), output_node_names.size());
    assert(output_tensors.size() == 3 && output_tensors.front().IsTensor());
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();     // 也可以使用output_tensors.front(); 获取list中的第一个元素变量  list.pop_front(); 删除list中的第一个位置的元素
    std::cout<<*floatarr<<std::endl;
    Mat newarr = Mat_<double>(1, 4); //定义一个1*1000的矩阵
    for (int i = 0; i < newarr.rows; i++)
    {
        for (int j = 0; j < newarr.cols; j++) //矩阵列数循环
        {
            newarr.at<double>(i, j) = floatarr[j];
        }
            }
    std::cout<<newarr<<std::endl;

    cv::rectangle(re_img,cv::Point2d(newarr.at<double>(0,0),newarr.at<double>(0,1)),cv::Point2d(newarr.at<double>(0,2),newarr.at<double>(0,3)),cv::Scalar(255,0,0),2);
    cv::imshow("output",re_img);
    cv::waitKey(0);
    return 0;
}
