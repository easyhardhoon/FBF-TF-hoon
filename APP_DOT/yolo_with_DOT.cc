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
#include "yolo_with_DOT.h"
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: ./minimal_yolo <tflite model>

using namespace std;

#define YOLO_INPUT "../../mAP_TF/input/images-optional/"
#define Partition_Num 7  // nCr --> "n"  // for YOLOv4-tiny
#define Max_Delegated_Partitions_Num 2  // nCr --> "r"  // hyper-param
#define GPU

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int combination(int n, int r) {
    	if(n == r || r == 0) return 1; 
    	else return combination(n - 1, r - 1) + combination(n - 1, r);
}

void read_image_opencv(string image_name, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(image_name, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, cv::COLOR_BGR2RGB); 
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(416,416));
	input.push_back(cvimg_);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  int DOT = combination(Partition_Num, Max_Delegated_Partitions_Num);
  vector<cv::Mat> input;
  for (int dot = 0; dot<DOT; dot++){
    int image_number = 1;
    printf("\n\n\033[0;31mDOT %d 's case starting...\033[0m\n\n", dot);
    for (int loop_num=0;loop_num<300;loop_num++){ 

      // Load image 
      std::string image_name = YOLO_INPUT + std::to_string(image_number) + ".jpg";
      read_image_opencv(image_name, input);

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

      #ifdef GPU
      // Modify interpreter::subgraph when using GPU
      TfLiteDelegate *MyDelegate = NULL;
      const TfLiteGpuDelegateOptionsV2 options = {
          .is_precision_loss_allowed = 0,  //1
          .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
          .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
          .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
          .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
          .priority_partition_num = dot, // loop_num
          .experimental_flags = 1,
          .max_delegated_partitions = Max_Delegated_Partitions_Num, // default is "1"
      };
      MyDelegate = TfLiteGpuDelegateV2Create(&options);
      TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(MyDelegate) == kTfLiteOk);
      #endif GPU

      // Allocate tensor buffers.
      TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
      printf("=== Pre-invoke Interpreter State ===\n");
      // tflite::PrintInterpreterState(interpreter.get());

      // Push image to input tensor
      auto input_tensor = interpreter->typed_input_tensor<float>(0);
      for (int i=0; i<416; i++){
        for (int j=0; j<416; j++){   
          cv::Vec3b pixel = input[0].at<cv::Vec3b>(i, j);
          *(input_tensor + i * 416*3 + j * 3) = ((float)pixel[0])/255.0;
          *(input_tensor + i * 416*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
          *(input_tensor + i * 416*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
        }
      }

      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
      printf("\n\n=== Interpreter Invoke ===\n");

      // Output parsing
      TfLiteTensor* cls_tensor = interpreter->output_tensor(1);
      TfLiteTensor* loc_tensor = interpreter->output_tensor(0);
      yolo_output_parsing(cls_tensor, loc_tensor);

      // Output visualize
      #ifndef GPU
      yolo_output_visualize(image_name, image_number);
      #endif

      // Make txt file to get mAP
      make_txt_to_get_mAP(yolo::YOLO_Parser::result_boxes, image_number, dot, Max_Delegated_Partitions_Num);

      // Re-initialize
      image_number+=1;
      input.clear();
      // interpreter->~Interpreter();
    }
  }
  cv::waitKey(0);
	cv::destroyAllWindows();
  return 0;
}
