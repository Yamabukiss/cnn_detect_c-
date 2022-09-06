#include "include/cnn_detect/detector.h"

const int resize_to=300;
const float threshold=0.5;
const int cpu_threads=16;
const char* model_path = "/home/yamabuki/Downloads/cnn_files/super_resolution.onnx";
const char* img_path="/home/yamabuki/Downloads/cnn_files/IMG_20181228_102706.jpg";

void img_process(cv::Mat& img)
{
    cv::resize(img,img,cv::Size(resize_to,resize_to));
    img/=255;
}

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(cpu_threads);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    clock_t start, stop;
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

    cv::Mat origin_img = cv::imread(img_path);
    cv::Mat img=origin_img.clone();
    img_process(img);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1, cv::Size(resize_to, resize_to));

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
    start = clock();
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
    int target_num=index_vec.size();

    std::vector<double> box_vec;
    for (int i = 0; i <target_num; i++)
    {
        box_vec.push_back(data_dim1[index_vec[i]]*origin_img.cols/resize_to);
        box_vec.push_back(data_dim1[index_vec[i]+1]*origin_img.rows/resize_to);
        box_vec.push_back(data_dim1[index_vec[i]+2]*origin_img.cols/resize_to);
        box_vec.push_back(data_dim1[index_vec[i]+3]*origin_img.rows/resize_to);
    }

    std::vector<int> label_vec;
    for (int i = 0; i <target_num; i++)
    {
        label_vec.push_back(data_dim2[index_vec[i]]);
    }

    for (int i=0;i<target_num*4;i+=4)
    {
        cv::rectangle(origin_img,cv::Point2d(box_vec[i],box_vec[i+1]),cv::Point2d(box_vec[i+2],box_vec[i+3]),cv::Scalar(255,0,0),2);
    }
    stop = clock();
    std::cout<<"time_cost:"<<(double)(stop - start) / CLOCKS_PER_SEC<<"ms"<<std::endl;
//    cv::imshow("output",origin_img);
//    cv::waitKey(0);
    return 0;
}
