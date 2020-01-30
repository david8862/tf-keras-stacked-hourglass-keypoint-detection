## C++ on-device (X86/ARM) inference app for Stacked Hourglass Networks

Here are some C++ implementation of the on-device inference for trained Stacked Hourglass Networks models, including forward propagation of the model, heatmap postprocess and coordinate rescale. Now we have 2 approaches with different inference engine for that:

* Tensorflow-Lite (verified on commit id: 1b8f5bc8011a1e85d7a110125c852a4f431d0f59)
* [MNN](https://github.com/alibaba/MNN) from Alibaba (not work)


### Tensorflow-Lite

1. Build TF-Lite lib

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
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
# cd tf-keras-stacked-hourglass-keypoint-detection/tools/
# python validate_hourglass.py --model_path=model.tflite --classes_path=../configs/coco_classes.txt --skeleton_path=../configs/coco_skeleton.txt --image_file=../example/sample.jpg --loop_count=5
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

