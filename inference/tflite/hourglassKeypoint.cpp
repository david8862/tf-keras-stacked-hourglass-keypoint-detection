//
//  hourglassKeypoint.cpp
//  Tensorflow-lite
//
//  Created by Xiaobin Zhang on 2020/01/19.
//

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <assert.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <algorithm>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include "hourglassKeypoint.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define LOG(x) std::cerr

#define HG_OUTPUT_STRIDE 4

namespace hourglassKeypoint {

// definition of a keypoint prediction record
typedef struct prediction {
    float x;
    float y;
    float confidence;
    int class_index;
}t_prediction;


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


// Hourglass postprocess for each prediction heat map
void hourglass_postprocess(const TfLiteTensor* feature_map, std::vector<t_prediction> &prediction_list, float heat_threshold)
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

    const float* data = reinterpret_cast<float*>(feature_map->data.raw);

    TfLiteIntArray* output_dims = feature_map->dims;

    int batch = output_dims->data[0];
    int height = output_dims->data[1];
    int width = output_dims->data[2];
    int channel = output_dims->data[3];

    auto unit = sizeof(float);

    // TF/TFLite tensor format: NHWC
    auto bytesPerRow   = channel * unit;
    auto bytesPerImage = width * bytesPerRow;
    auto bytesPerBatch = height * bytesPerImage;

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        LOG(INFO) << "batch " << b << "\n";

        for (int c = 0; c < channel; c++) {
            float max_heatmap_value = 0.0;
            int max_heatmap_x = 0;
            int max_heatmap_y = 0;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // TF/TFLite tensor format: NHWC
                    int heatmap_offset = h * width * channel + w * channel + c;

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
    assert(input_width == input_height);

    float scale_width = float(image_width) / float(input_width);
    float scale_height = float(image_height) / float(input_height);


    for(auto &prediction : prediction_list) {
        prediction.x = prediction.x * scale_width * HG_OUTPUT_STRIDE;
        prediction.y = prediction.y * scale_height * HG_OUTPUT_STRIDE;
    }

    return;
}


template <typename T>
void resize(T* out, uint8_t* in, int image_width, int image_height,
            int image_channels, int wanted_width, int wanted_height,
            int wanted_channels, Settings* s) {
  uint8_t* resized = (uint8_t*)malloc(wanted_height * wanted_width * wanted_channels * sizeof(uint8_t));
  if (resized == nullptr) {
      LOG(FATAL) << "Can't alloc memory" << "\n";
      exit(-1);
  }

  stbir_resize_uint8(in, image_width, image_height, 0,
                     resized, wanted_width, wanted_height, 0, wanted_channels);

  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (resized[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)resized[i];
  }

  free(resized);
  return;
}


void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  // load model
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  s->model = model.get();
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  // prepare model interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);
  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  // get classes labels
  std::vector<std::string> classes;
  std::ifstream classesOs(s->classes_file_name.c_str());
  std::string line;
  while (std::getline(classesOs, line)) {
      classes.emplace_back(line);
  }
  int num_classes = classes.size();
  LOG(INFO) << "num_classes: " << num_classes << "\n";

  // read input image
  int image_width = 224;
  int image_height = 224;
  int image_channel = 3;

  auto input_image = (uint8_t*)stbi_load(s->input_img_name.c_str(), &image_width, &image_height, &image_channel, 3);
  if (input_image == nullptr) {
      LOG(FATAL) << "Can't open" << s->input_img_name << "\n";
      exit(-1);
  }

  std::vector<uint8_t> in(input_image, input_image + image_height * image_width * image_channel * sizeof(uint8_t));

  // free input image
  stbi_image_free(input_image);
  input_image = nullptr;

  // assuming one input only
  int input = interpreter->inputs()[0];

  LOG(INFO) << "origin image size: width:" << image_width
            << ", height:" << image_height
            << ", channel:" << image_channel
            << "\n";

  // get input dimension from the input tensor metadata
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  // check input dimension
  assert(dims->size == 4);

  int input_batch = dims->data[0];
  int input_height = dims->data[1];
  int input_width = dims->data[2];
  int input_channels = dims->data[3];

  std::vector<std::string> tensor_type_string = {"kTfLiteNoType",
                                                 "kTfLiteFloat32",
                                                 "kTfLiteInt32",
                                                 "kTfLiteUInt8",
                                                 "kTfLiteInt64",
                                                 "kTfLiteString",
                                                 "kTfLiteBool",
                                                 "kTfLiteInt16",
                                                 "kTfLiteComplex64",
                                                 "kTfLiteInt8",
                                                 "kTfLiteFloat16",
                                                 "kTfLiteFloat64",
                                                 "kTfLiteComplex128",
                                                 "kTfLiteUInt64",
                                                 "kTfLiteResource",
                                                 "kTfLiteVariant",
                                                 "kTfLiteUInt32"
                                                };

  if (s->verbose) LOG(INFO) << "input tensor info: "
                            << "name " << interpreter->tensor(input)->name << ", "
                            << "type " << tensor_type_string[interpreter->tensor(input)->type] << ", "
                            << "dim_size " << interpreter->tensor(input)->dims->size << ", "
                            << "batch " << input_batch << ", "
                            << "height " << input_height << ", "
                            << "width " << input_width << ", "
                            << "channels " << input_channels << "\n";

  // assume the model input is square
  assert(input_width == input_height);

  // resize image to model input shape
  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      s->input_floating = true;
      resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_width, image_height, image_channel, input_width,
                    input_height, input_channels, s);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      image_width, image_height, image_channel, input_width,
                      input_height, input_channels, s);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }


  // run warm up session
  if (s->loop_count > 1)
    for (int i = 0; i < s->number_of_warmup_runs; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }

  // run model sessions to get output
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < s->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked average time:" << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000) << " ms \n";


  // Do hourglass_postprocess to parse out valid predictions
  std::vector<t_prediction> prediction_list;
  float heat_threshold = 1e-6;
  const std::vector<int> outputs = interpreter->outputs();

  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < outputs.size(); i++) {
      int output = interpreter->outputs()[i];
      TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
      // check output dimension
      assert(output_dims->size == 4);

      int output_batch = output_dims->data[0];
      int output_height = output_dims->data[1];
      int output_width = output_dims->data[2];
      int output_channels = output_dims->data[3];

      // output channel should be same as
      // keypoint class number
      assert(num_classes == output_channels);

      // input/output shape should match hourglass output stride
      assert((input_width/output_width == HG_OUTPUT_STRIDE) && (input_height/output_height == HG_OUTPUT_STRIDE));

      if (s->verbose) LOG(INFO) << "output tensor info: "
                                << "name " << interpreter->tensor(output)->name << ", "
                                << "type " << tensor_type_string[interpreter->tensor(output)->type] << ", "
                                << "dim_size " << interpreter->tensor(output)->dims->size << ", "
                                << "batch " << output_batch << ", "
                                << "height " << output_height << ", "
                                << "width " << output_width << ", "
                                << "channels " << output_channels << "\n";
      TfLiteTensor* feature_map = interpreter->tensor(output);

      hourglass_postprocess(feature_map, prediction_list, heat_threshold);
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "hourglass_postprocess time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms\n";
  LOG(INFO) << "prediction_list length: " << prediction_list.size() << "\n";

  // Rescale the prediction back to original image
  adjust_scale(prediction_list, image_width, image_height, input_width, input_height);

  // Show detection result
  LOG(INFO) << "Keypoint Detection result:\n";
  for(auto prediction : prediction_list) {
      LOG(INFO) << classes[prediction.class_index] << " "
                << prediction.confidence << " "
                << "(" << int(prediction.x) << ", " << int(prediction.y) << ")\n";
  }

  return;
}

void display_usage() {
  LOG(INFO)
      << "Usage: hourglassKeypoint\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--image, -i: image_name.jpg\n"
      << "--classes, -l: classes labels for the model\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--threads, -t: number of threads\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"tflite_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:f:hi:l:m:s:t:v:w:", long_options,
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
      case 'f':
        s.allow_fp16 =
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
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
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
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace hourglassKeypoint

int main(int argc, char** argv) {
  return hourglassKeypoint::Main(argc, argv);
}
