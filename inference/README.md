## C++ on-device (X86/ARM) inference app for Stacked Hourglass Networks

Here are some C++ implementation of the on-device inference for trained Stacked Hourglass Networks models, including forward propagation of the model, heatmap postprocess and coordinate rescale. Now we have 2 approaches with different inference engine for that:

* Tensorflow-Lite (verified on tag: v2.6.0)
* [MNN](https://github.com/alibaba/MNN) from Alibaba (verified on release: [1.0.0](https://github.com/alibaba/MNN/releases/tag/1.0.0))


### MNN

1. Install Python runtime and Build libMNN

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux), we need to prepare cmake & protobuf first for MNN build. And since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile
```
# apt install cmake autoconf automake libtool ocl-icd-opencl-dev
# wget https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-cpp-3.4.1.tar.gz
# tar xzvf protobuf-cpp-3.4.1.tar.gz
# cd protobuf-3.4.1
# ./autogen.sh
# ./configure && make && make check && make install && ldconfig
# pip install --upgrade pip && pip install --upgrade mnn

# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TRAIN=ON -MNN_BUILD_TRAIN_MINI=ON -MNN_USE_OPENCV=OFF] ..
        && make -j4

### MNN OpenCL backend build
# apt install ocl-icd-opencl-dev
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_USE_SYSTEM_LIB=ON] ..
        && make -j4
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" should be specified

"MNN_BUILD_QUANTOOLS" is for enabling MNN Quantization tool

"MNN_BUILD_CONVERTER" is for enabling MNN model converter

"MNN_BUILD_BENCHMARK" is for enabling on-device inference benchmark tool

"MNN_BUILD_TRAIN" related are for enabling MNN training tools


2. Build demo inference application
```
# cd tf-keras-stacked-hourglass-keypoint-detection/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

3. Convert trained Stacked Hourglass model to MNN model

Refer to [Model dump](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection#model-dump), [Tensorflow model convert](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection#tensorflow-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we need to:

* dump out inference model from training checkpoint:

    ```
    # python demo.py --num_stacks=2 --mobile --weights_path=logs/<checkpoint>.h5 --classes_path=configs/coco_classes.txt --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to tensorflow frozen pb model:

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

* convert TF pb model to MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./MNNConvert -f TF --modelFile model.pb --MNNModel model.pb.mnn --bizCode MNN
    ```
    or

    ```
    # mnnconvert -f TF --modelFile model.pb --MNNModel model.pb.mnn
    ```

MNN support Post Training Integer quantization, so we can use its python CLI interface to do quantization on the generated .mnn model to get quantized .mnn model for ARM acceleration . A json config file [quantizeConfig.json](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/inference/MNN/configs/quantizeConfig.json) is needed to describe the feeding data:

* Quantized MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./quantized.out model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```
    or

    ```
    # mnnquant model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```

4. Run validate script to check MNN model
```
# cd tf-keras-stacked-hourglass-keypoint-detection/tools/evaluation/
# python validate_hourglass.py --model_path=model_quant.pb.mnn --classes_path=../../configs/coco_classes.txt --skeleton_path=../../configs/coco_skeleton.txt --image_file=../../example/fitness.jpg --model_input_shape=256x256 --loop_count=5
```

Visualized detection result:

<p align="center">
  <img src="../assets/dog_inference.jpg">
</p>

#### You can also use [eval.py](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection#evaluation) to do evaluation on the MNN model


5. Run application to do inference with model, or put all the assets to your ARM board and run if you use cross-compile
```
# cd tf-keras-stacked-hourglass-keypoint-detection/inference/MNN/build
# ./hourglassKeypoint -h
Usage: hourglassKeypoint
--mnn_model, -m: model_name.mnn
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--threads, -t: number of threads
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs

# ./hourglassKeypoint -m model.pb.mnn -i ../../../example/fitness.jpg -l ../../../configs/coco_classes.txt -t 8 -c 10 -w 3
image_input: name:image_input, width:256, height:256, channel:3, dim_type:CAFFE
num_outputs: 1
num_classes: 17
origin image size: width:640, height:640, channel:3
model invoke average time: 170.583000 ms
output tensor name: 1_conv_1x1_parts/BiasAdd
output tensor shape: width:64 , height:64, channel: 17
heatmap shape: batch:1, width:64 , height:64, channel: 17
Caffe format: NCHW
batch 0:
hourglass_postprocess time: 0.105000 ms
prediction_list length: 17
Keypoint Detection result:
nose 0.283470 (310, 70)
left_eye 0.419499 (310, 60)
right_eye 0.333521 (290, 60)
left_ear 0.303642 (330, 60)
right_ear 0.046196 (270, 70)
left_shoulder 0.637282 (380, 140)
right_shoulder 0.740778 (260, 150)
left_elbow 0.740789 (410, 220)
right_elbow 0.678265 (240, 240)
left_wrist 0.863041 (420, 300)
right_wrist 0.549483 (240, 310)
left_hip 0.532070 (350, 330)
right_hip 0.475801 (290, 330)
left_knee 0.704592 (380, 450)
right_knee 0.748527 (280, 450)
left_ankle 0.570860 (350, 570)
right_ankle 0.596960 (280, 590)
```
Here the [classes](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/configs/mpii_classes.txt) file format are the same as used in training part



### Tensorflow-Lite

1. Build TF-Lite lib

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# git checkout v2.6.0
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
```

you can also create your own build script for new ARM platform, like:

```shell
# vim ./tensorflow/lite/tools/make/build_my_arm_lib.sh


#!/bin/bash -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."

make CC_PREFIX=/root/toolchain/aarch64-linux-gnu/bin/aarch64-linux-gnu- -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=myarm TARGET_ARCH=aarch64 $@
```

**NOTE:**
* Using Makefile to build TensorFlow Lite is deprecated since Aug 2021. So v2.6.0 should be the last major version to support Makefile build (cmake is enabled on new version)
* by default TF-Lite build only generate static lib (.a), but we can do minor change in Makefile to generate .so shared lib together, as follow:

```diff
diff --git a/tensorflow/lite/tools/make/Makefile b/tensorflow/lite/tools/make/Makefile
index 662c6bb5129..83219a42845 100644
--- a/tensorflow/lite/tools/make/Makefile
+++ b/tensorflow/lite/tools/make/Makefile
@@ -99,6 +99,7 @@ endif
 # This library is the main target for this makefile. It will contain a minimal
 # runtime that can be linked in to other programs.
 LIB_NAME := libtensorflow-lite.a
+SHARED_LIB_NAME := libtensorflow-lite.so

 # Benchmark static library and binary
 BENCHMARK_LIB_NAME := benchmark-lib.a
@@ -301,6 +302,7 @@ BINDIR := $(GENDIR)bin/
 LIBDIR := $(GENDIR)lib/

 LIB_PATH := $(LIBDIR)$(LIB_NAME)
+SHARED_LIB_PATH := $(LIBDIR)$(SHARED_LIB_NAME)
 BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
 BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)
 BENCHMARK_PERF_OPTIONS_BINARY := $(BINDIR)$(BENCHMARK_PERF_OPTIONS_BINARY_NAME)
@@ -344,7 +346,7 @@ $(OBJDIR)%.o: %.c
        $(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

 # The target that's compiled if there's no command-line arguments.
-all: $(LIB_PATH)  $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)
+all: $(LIB_PATH) $(SHARED_LIB_PATH) $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

 # The target that's compiled for micro-controllers
 micro: $(LIB_PATH)
@@ -361,7 +363,14 @@ $(LIB_PATH): tensorflow/lite/experimental/acceleration/configuration/configurati
        @mkdir -p $(dir $@)
        $(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

-lib: $(LIB_PATH)
+$(SHARED_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(LIB_OBJS)
+       @mkdir -p $(dir $@)
+       $(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB_PATH) $(LIB_OBJS)
+$(SHARED_LIB_PATH): tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h $(LIB_OBJS)
+       @mkdir -p $(dir $@)
+       $(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB_PATH) $(LIB_OBJS)
+
+lib: $(LIB_PATH) $(SHARED_LIB_PATH)

 $(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
        @mkdir -p $(dir $@)
```


2. Build demo inference application
```
# cd tf-keras-stacked-hourglass-keypoint-detection/inference/tflite
# mkdir build && cd build
# cmake -DTF_ROOT_PATH=<Path_to_TF> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] [-DTARGET_PLAT=<target>] ..
# make
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" and "TARGET_PLAT" should be specified. Refer [CMakeLists.txt](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/inference/tflite/CMakeLists.txt) for details.

3. Convert trained Stacked Hourglass model to tflite model

Currently, We only support dumping out keras .h5 model to Float32 .tflite model:

* dump out inference model from training checkpoint:

    ```
    # python demo.py --num_stacks=2 --mobile --weights_path=logs/<checkpoint>.h5 --classes_path=configs/coco_classes.txt --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to Float32 tflite model:

    ```
    # tflite_convert --keras_model_file=model.h5 --output_file=model.tflite
    ```

4. Run validate script to check TFLite model
```
# cd tf-keras-stacked-hourglass-keypoint-detection/tools/evaluation/
# python validate_hourglass.py --model_path=model.tflite --classes_path=../../configs/coco_classes.txt --skeleton_path=../../configs/coco_skeleton.txt --image_file=../../example/sample.jpg --loop_count=5
```

5. Run application to do inference with model, or put assets to ARM board and run if cross-compile
```
# cd tf-keras-stacked-hourglass-keypoint-detection/inference/tflite/build
# ./hourglassKeypoint -h
Usage: hourglassKeypoint
--tflite_model, -m: model_name.tflite
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./hourglassKeypoint -m model.tflite -i ../../../example/sample.jpg -l ../../../configs/coco_classes.txt -t 8 -c 10 -w 3 -v 1
resolved reporter
num_classes: 17
origin image size: width:1080, height:1440, channel:3
input tensor info: type 1, batch 1, height 256, width 256, channels 3
invoked average time:305.514 ms
output tensor info: name 1_conv_1x1_parts/BiasAdd, type 1, batch 1, height 64, width 64, channels 17
batch 0
hourglass_postprocess time: 1.655 ms
prediction_list length: 17
Keypoint Detection result:
nose 0.7050172 (473, 180)
left_eye 0.67753065 (489, 135)
right_eye 0.6072198 (456, 158)
left_ear 0.65866643 (557, 135)
right_ear 0.0035909468 (405, 158)
left_shoulder 0.5254785 (658, 293)
right_shoulder 0.43146 (388, 338)
left_elbow 0.6435836 (692, 495)
right_elbow 0.6751519 (371, 563)
left_wrist 0.66792876 (726, 675)
right_wrist 0.61613876 (456, 720)
left_hip 0.5049588 (641, 743)
right_hip 0.35920674 (489, 743)
left_knee 0.62980515 (726, 990)
right_knee 0.53673136 (506, 1035)
left_ankle 0.57830113 (574, 1260)
right_ankle 0.58238965 (489, 1305)
```

### TODO
- [ ] support letterbox input image in validate script and C++ inference code

