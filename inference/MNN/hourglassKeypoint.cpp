//
//  hourglassKeypoint.cpp
//  MNN
//
//  Created by Xiaobin Zhang on 2020/01/17.
//

#include <stdio.h>
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define HG_OUTPUT_STRIDE 4

using namespace MNN;
using namespace MNN::CV;


// definition of a keypoint prediction record
typedef struct prediction {
    float x;
    float y;
    float confidence;
    int class_index;
}t_prediction;


// model inference settings
struct Settings {
  int loop_count = 1;
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
  float input_mean = 127.5f;
  float input_std = 255.0f;
  std::string model_name = "./model.mnn";
  std::string input_img_name = "./sample.jpg";
  std::string classes_file_name = "./classes.txt";
  bool input_floating = false;
  //bool verbose = false;
  //string input_layer_type = "uint8_t";
};


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


void display_usage() {
    std::cout
        << "Usage: hourglassKeypoint\n"
        << "--mnn_model, -m: model_name.mnn\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
        << "--threads, -t: number of threads\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        //<< "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


// Hourglass postprocess for each prediction heat map
void hourglass_postprocess(const Tensor* heatmap, std::vector<t_prediction> &prediction_list, float heat_threshold)
{
    // Do following transform to get the precision
    // keypoints:
    //
    // 1. filter heatmap value under threshold
    //
    //    map[map < threshold] = 0
    //
    // 2. choose peak point in heatmap and get its
    //    coordinate & confidence as keypoint:
    //
    //    y, x = np.where(map == map.max())
    //
    // 3. enqueue the keypoint info if exists:
    //
    //    if len(x) > 0 and len(y) > 0:
    //        kplst.append((int(x[0]), int(y[0]), _map[y[0], x[0]]))

    const float* data = heatmap->host<float>();
    auto dimType = heatmap->getDimensionType();

    auto batch   = heatmap->batch();
    auto channel = heatmap->channel();
    auto height  = heatmap->height();
    auto width   = heatmap->width();

    MNN_PRINT("heatmap shape: batch:%d, width:%d , height:%d, channel: %d\n", batch, width, height, channel);

    auto unit = sizeof(float);

    // now we only support single image postprocess
    MNN_ASSERT(batch == 1);

    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");

        bytesPerRow   = channel * unit;
        bytesPerImage = width * bytesPerRow;
        bytesPerBatch = height * bytesPerImage;

    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");

        bytesPerRow   = width * unit;
        bytesPerImage = height * bytesPerRow;
        bytesPerBatch = channel * bytesPerImage;

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        MNN_PRINT("batch %d:\n", b);

        for (int c = 0; c < channel; c++) {
            float max_heatmap_value = 0.0;
            int max_heatmap_x = 0;
            int max_heatmap_y = 0;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int heatmap_offset;
                    if (dimType == Tensor::TENSORFLOW) {
                        // Tensorflow format tensor, NHWC
                        heatmap_offset = h * width * channel + w * channel + c;
                    } else if (dimType == Tensor::CAFFE) {
                        // Caffe format tensor, NCHW
                        heatmap_offset = c * width * height + h * width + w;
                    } else if (dimType == Tensor::CAFFE_C4) {
                        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
                        exit(-1);
                    } else {
                        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
                        exit(-1);
                    }

                    float heatmap_value = bytes[heatmap_offset];
                    if (heatmap_value < heat_threshold) {
                        // filter under threshold
                        heatmap_value = 0;
                    }
                    if(heatmap_value > max_heatmap_value) {
                        // got a peak heatmap value, update the peak record
                        max_heatmap_value = heatmap_value;
                        max_heatmap_x = w;
                        max_heatmap_y = h;
                    }
                }
            }

            if(max_heatmap_x > 0 && max_heatmap_y > 0) {
                // got a valid prediction, form up data and push to result vector
                t_prediction keypoint_prediction;
                keypoint_prediction.x = max_heatmap_x;
                keypoint_prediction.y = max_heatmap_y;
                keypoint_prediction.confidence = max_heatmap_value;
                keypoint_prediction.class_index = c;

                prediction_list.emplace_back(keypoint_prediction);
            }
        }
    }
    return;
}


void adjust_scale(std::vector<t_prediction> &prediction_list, int image_width, int image_height, int input_width, int input_height)
{
    // Rescale the final prediction back to original image
    MNN_ASSERT(input_width == input_height);

    float scale_width = float(image_width) / float(input_width);
    float scale_height = float(image_height) / float(input_height);


    for(auto &prediction : prediction_list) {
        prediction.x = prediction.x * scale_width * HG_OUTPUT_STRIDE;
        prediction.y = prediction.y * scale_height * HG_OUTPUT_STRIDE;
    }

    return;
}


// Resize image to model input shape
uint8_t* image_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    MNN_ASSERT(image_channel == input_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
        exit(-1);
    }
    stbir_resize_uint8(inputImage, image_width, image_height, 0,
                     input_image, input_width, input_height, 0, image_channel);

    return input_image;
}


template <class T>
void fill_data(T* out, uint8_t* in, int input_width, int input_height,
            int input_channels, Settings* s) {
  auto output_number_of_pixels = input_height * input_width * input_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (in[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)in[i];
  }

  return;
}


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;

    // create model & session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(s->model_name.c_str()));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO; //MNN_FORWARD_CPU, MNN_FORWARD_OPENCL
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = s->number_of_threads;

    BackendConfig bnconfig;
    bnconfig.memory = BackendConfig::Memory_Normal; //Memory_High, Memory_Low
    bnconfig.power = BackendConfig::Power_Normal; //Power_High, Power_Low
    bnconfig.precision = BackendConfig::Precision_Normal; //Precision_High, Precision_Low
    config.backendConfig = &bnconfig;

    auto session = net->createSession(config);
    // since we don't need to create other sessions any more,
    // just release model data to save memory
    net->releaseModel();

    // get input tensor info
    // assume only 1 input tensor (image_input)
    auto inputs = net->getSessionInputAll(session);
    MNN_ASSERT(inputs.size() == 1);
    auto image_input = inputs.begin()->second;

    int input_width = image_input->width();
    int input_height = image_input->height();
    int input_channel = image_input->channel();
    int input_dim_type = image_input->getDimensionType();

    std::vector<std::string> dim_type_string = {"TENSORFLOW", "CAFFE", "CAFFE_C4"};

    MNN_PRINT("image_input: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", inputs.begin()->first.c_str(), input_width, input_height, input_channel, dim_type_string[input_dim_type].c_str());

    // assume the model input is square
    MNN_ASSERT(input_width == input_height);

    //auto shape = image_input->shape();
    //shape[0] = 1;
    //net->resizeTensor(image_input, shape);
    //net->resizeSession(session);

    // get output tensor info:
    // image_input: 1 x 256 x 256 x 3
    // "1_conv_1x1_parts/Conv2D": 1 x 64 x 64 x (num_classes)
    auto outputs = net->getSessionOutputAll(session);
    int num_outputs = outputs.size();
    MNN_PRINT("num_outputs: %d\n", num_outputs);

    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    MNN_PRINT("num_classes: %d\n", num_classes);

    // load input image
    auto inputPath = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    uint8_t* inputImage = (uint8_t*)stbi_load(inputPath, &image_width, &image_height, &image_channel, input_channel);
    if (nullptr == inputImage) {
        MNN_ERROR("Can't open %s\n", inputPath);
        return;
    }
    MNN_PRINT("origin image size: width:%d, height:%d, channel:%d\n", image_width, image_height, image_channel);

    // resize input image
    uint8_t* resizeImage = image_resize(inputImage, image_width, image_height, image_channel, input_width, input_height, input_channel);

    // free input image
    stbi_image_free(inputImage);
    inputImage = nullptr;

    // assume input tensor type is float
    MNN_ASSERT(image_input->getType().code == halide_type_float);
    s->input_floating = true;

    // create a host tensor for input data
    auto dataTensor = new Tensor(image_input, Tensor::TENSORFLOW);
    fill_data<float>(dataTensor->host<float>(), resizeImage,
                input_width, input_height, input_channel, s);

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            image_input->copyFromHostTensor(dataTensor);
            if (net->runSession(session) != NO_ERROR) {
                MNN_PRINT("Failed to invoke MNN!\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        image_input->copyFromHostTensor(dataTensor);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));


    // Copy output tensors to host, for further postprocess
    std::vector<std::shared_ptr<Tensor>> heatmaps;
    for(auto output : outputs) {
        MNN_PRINT("output tensor name: %s\n", output.first.c_str());
        auto output_tensor = output.second;

        int output_width = output_tensor->width();
        int output_height = output_tensor->height();
        int output_channel = output_tensor->channel();
        MNN_PRINT("output tensor shape: width:%d , height:%d, channel: %d\n", output_width, output_height, output_channel);

        // output channel should be same as
        // keypoint class number
        MNN_ASSERT(num_classes == output_channel);

        // input/output shape should match hourglass output stride
        MNN_ASSERT((input_width/output_width == HG_OUTPUT_STRIDE) && (input_height/output_height == HG_OUTPUT_STRIDE));

        auto dim_type = output_tensor->getDimensionType();
        if (output_tensor->getType().code != halide_type_float) {
            dim_type = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> output_user(new Tensor(output_tensor, dim_type));
        output_tensor->copyToHostTensor(output_user.get());
        heatmaps.emplace_back(output_user);
    }

    // Do hourglass_postprocess to parse out valid predictions
    std::vector<t_prediction> prediction_list;
    float heat_threshold = 1e-6;

    gettimeofday(&start_time, nullptr);

    for (int i = 0; i < num_outputs; ++i) {
        Tensor* heatmap = heatmaps[i].get();

        // Now we only support float32 type output tensor
        MNN_ASSERT(heatmap->getType().code == halide_type_float);
        MNN_ASSERT(heatmap->getType().bits == 32);
        hourglass_postprocess(heatmap, prediction_list, heat_threshold);
    }

    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("hourglass_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);
    MNN_PRINT("prediction_list length: %lu\n", prediction_list.size());

    // Rescale the prediction back to original image
    adjust_scale(prediction_list, image_width, image_height, input_width, input_height);

    // Show detection result
    MNN_PRINT("Keypoint Detection result:\n");
    for(auto prediction : prediction_list) {
        MNN_PRINT("%s %f (%d, %d)\n", classes[prediction.class_index].c_str(), prediction.confidence, int(prediction.x), int(prediction.y));
    }

    // Release buffer memory
    if (resizeImage) {
        free(resizeImage);
        resizeImage = nullptr;
    }

    delete dataTensor;

    // Release session and model
    net->releaseSession(session);
    //net->releaseModel();

    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"mnn_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        //{"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:hi:l:m:s:t:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_img_name = optarg;
        break;
      case 'l':
        s.classes_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      //case 'v':
        //s.verbose =
            //strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        //break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
      default:
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

