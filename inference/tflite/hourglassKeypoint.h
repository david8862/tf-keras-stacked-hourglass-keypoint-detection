//
//  hourglassKeypoint.h
//  Tensorflow-lite
//
//  Created by Xiaobin Zhang on 2020/01/19.
//
//

#ifndef HOURGLASS_KEYPOINT_HOURGLASS_KEYPOINT_H_
#define HOURGLASS_KEYPOINT_HOURGLASS_KEYPOINT_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace hourglassKeypoint {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool input_floating = false;
  bool allow_fp16 = false;
  int loop_count = 1;
  float input_mean = 0.0f;
  float input_std = 255.0f;
  std::string model_name = "./model.tflite";
  tflite::FlatBufferModel* model;
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
};

}  // namespace hourglassKeypoint

#endif  // HOURGLASS_KEYPOINT_HOURGLASS_KEYPOINT_H_
