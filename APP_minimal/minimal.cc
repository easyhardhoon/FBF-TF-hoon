/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <chrono>
#include "minimal.h"
#define Use_GPU
#define DEBUG
// #define YOLO
// #include "minimal_yolo.h"
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: ./minimal_yolo <tflite model>
// -----------------------------------------------------------------------
// NOTE : APP_vaniila for debugging each tflite model's architecture , 
//        delegated_partitions, execution_plan
//        should use default "FirstNLargest" delegation implement function
// -----------------------------------------------------------------------
using namespace std;

#define INPUT "../../mAP_TF/input/images-optional/"
#define IMAGE_NUM 300
#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

uint64_t millis() {
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    return ms; 
}

void read_image_opencv(string image_name, vector<cv::Mat>& input, int width, int height){
	cv::Mat cvimg = cv::imread(image_name, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, cv::COLOR_BGR2RGB); 
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(width, height));
	input.push_back(cvimg_);
}

std::map<std::string, std::pair<int,int>> input_map = {
  {"yolo", {416,416}},
  {"lane", {512,256}},
  {"move", {192,192}}
};

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* input_type = argv[2];
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  #ifdef Use_GPU
  // Modify interpreter::subgraph when using GPU
  TfLiteDelegate *MyDelegate = NULL;
  const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 0,  //1
      .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
      .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
      .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .priority_partition_num = 1,
      .experimental_flags = 1,
      .max_delegated_partitions = 10, 
  };
  MyDelegate = TfLiteGpuDelegateV2Create(&options);
  TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(MyDelegate) == kTfLiteOk);
  #endif

  printf("=== Print interpreterstate Start ===\n");
  #ifdef DEBUG
  tflite::PrintInterpreterState(interpreter.get());
  #endif
  printf("=== Print interpreterstate End ===\n");
  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());
  vector<cv::Mat> input;
  for (int i=1;i<=IMAGE_NUM;i++){
    
    // Load image 
    std::cout << "\033[0;35mNew LOOP Start\033[0m" <<std::endl;
    std::string image_name = INPUT + std::to_string(i) + ".jpg";
    int width = input_map[input_type].first;
    int height = input_map[input_type].second;
    read_image_opencv(image_name, input, width, height); 

    // Push image to input tensor 
    // TFLite's data tensor format :[B, H, W, C]
    // Opencv's data image  format :[W, H] 
    auto input_tensor = interpreter->typed_input_tensor<float>(0);

    // ISSUE (SOLVED)
    // for (int h=0; h<height; h++){
    //   for (int w=0; w<width; w++){
    //     cv::Vec3b pixel = input[0].at<cv::Vec3b>(w, h);
    //     *(input_tensor + h * width*3 + w * 3) = ((float)pixel[0])/255.0;
    //     *(input_tensor + h * width*3 + w * 3 + 1) = ((float)pixel[1])/255.0;
    //     *(input_tensor + h * width*3 + w * 3 + 2) = ((float)pixel[2])/255.0;
    //   }
    // }

    // CORRECT
    for (int i=0; i<416; i++){
               for (int j=0; j<416; j++){
                 cv::Vec3b pixel = input[0].at<cv::Vec3b>(i, j);
                 *(input_tensor + i * 416*3 + j * 3) = ((float)pixel[0])/255.0;
                 *(input_tensor + i * 416*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
                 *(input_tensor + i * 416*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
               }
             }
    // Run inference
    printf("\n=== Invoke START ===\n");
    uint64_t START = millis();
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    uint64_t END = millis();
    uint64_t Invoke_time = END - START;
    std::cout << "\nInvoke_time : " << Invoke_time << "ms" << std::endl;
    printf("=== Invoke END ===\n");

    #ifdef YOLO
    // Output parsing
    TfLiteTensor* cls_tensor = interpreter->output_tensor(1);
    TfLiteTensor* loc_tensor = interpreter->output_tensor(0);
    yolo_output_parsing(cls_tensor, loc_tensor);
    make_txt_to_get_mAP(yolo::YOLO_Parser::result_boxes, i);
    // Output visualize
    // yolo_output_visualize(image_name, i);
    #endif
    input.clear();
  }  
  // cv::waitKey(0);
	// cv::destroyAllWindows();
  return 0;
}
