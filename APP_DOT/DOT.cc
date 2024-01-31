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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "DOT.h"
#include <chrono>
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.

// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: ./minimal_yolo <tflite model>

using namespace std;

#define INPUT "../../mAP_TF/input/images-optional/"
#define Partition_Num 9  // nCr --> "n"  // for YOLOv4-tiny
// #define Max_Delegated_Partitions_Num 1  // nCr --> "r"  // hyper-param // Not use in full-auto
#define GPU
#define IMG_set_num 1 // "300" for mAP , "100" for DOT // "1" for debugging
// #define YOLO

std::vector<float> time_table;
std::vector<std::vector<float>> DOT_table;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
}

void print_DOT_table(std::vector<std::vector<float>> DOT_table){
  std::cout << "\033[0;31m////////////////////////////////////////////////////////\033[0m" <<std::endl;
  std::cout << "\033[0;31m////////////////////////////////////////////////////////\033[0m" <<std::endl;
  std::cout << "\033[0;31m/////Print DOT Table//////\033[0m" <<std::endl;
  for (int i=0;i<DOT_table.size();i++){
    std::cout << "N = " << i+1 <<std::endl;
    std::vector<float> time_table = DOT_table[i];
    float min = *min_element(time_table.begin(), time_table.end());
    float bias = 0.3;
    float sum = 0.0;
    for (int j=0;j< time_table.size(); j++){
        printCombination(Partition_Num,i+1,j);
        if(time_table[j] < min + bias) {
            printf("\033[0;31mcase's latency is %0.2fms\033[0m\n",time_table[j]);
        }
        else{
            std::cout << "case's latency is " << time_table[j] << "ms" << std::endl;
        }
        sum += time_table[j];
    }
    printf("\033[0;32m[END]...Choose_%d's average latency is %0.2fms\033[0m\n",i+1,sum/time_table.size());
  }
};

void find_best_case(std::vector<std::vector<float>> DOT_table){
  float min_value = DOT_table[0][0];
  int min_row = 0;
  int min_col = 0;
  for (int i = 0; i < DOT_table.size(); ++i) {
      for (int j = 0; j < DOT_table[i].size(); ++j) {
          if (DOT_table[i][j] < min_value) {
              min_value = DOT_table[i][j];
              min_row = i;
              min_col = j;
          }
      }
  }
  printf("Minimum CAES's value : %.2fms\n", min_value);
  printf("Minimum CASE's N : %d\n",min_row+1);
  printf("Minimum CASE's th : %d\n",min_col);
  printf("Minimum CASE's combination : ");
  printCombination(Partition_Num, min_row+1,min_col);
  printf("\n");
};

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <input_type> \n");
    return 1;
  }
  const char* filename = argv[1];
  const char* input_type = argv[2];
  vector<cv::Mat> input;
  for (int N=1;N<=Partition_Num;N++){
  //////////////////////////////////////////////////////////////////////////////////////////
  // Outer loop [1<=N<=Max] 
    printf("\033[0;33mLOOP START [N=%d]...\033[0m\n", N);
    int DOT = combination(Partition_Num, N);
    for (int dot = 0; dot<DOT; dot++){
        //////////////////////////////////////////////////////////////////////////////////////////
        // Build interpreter on each dot case
        int image_number = 1;
        uint64_t average_time = 0;
        printf("\033[0;32mDOT %d 's case starting...\033[0m\n", dot);

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
            .max_delegated_partitions = N, // default is "1"
        };
        MyDelegate = TfLiteGpuDelegateV2Create(&options);
        TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(MyDelegate) == kTfLiteOk);
        #endif GPU

        // Allocate tensor buffers.
        TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
        printf("=== Pre-invoke Interpreter State ===\n");
	      // tflite::PrintInterpreterState(interpreter.get());  //For debugging model info
        //////////////////////////////////////////////////////////////////////////////////////////
        // Push test image to input_tensor
        // float*tmp = nullptr;
        for (int loop_num=0;loop_num<IMG_set_num;loop_num++){ 
          // Load image 
          #ifdef YOLO
          std::string image_name = INPUT + std::to_string(image_number) + ".jpg";
          #endif
          #ifndef YOLO
          std::string image_name = INPUT + std::to_string(0) + ".jpg";
          #endif
          int width = input_map[input_type].first;
          int height = input_map[input_type].second;
          read_image_opencv(image_name, input, width, height); 
          printf("\n\n=== read image by CV (After)===\n");
          // Push image to input tensor
          auto input_tensor = interpreter->typed_input_tensor<float>(0); // float * , data.raw
          auto input_T = interpreter->input_tensor(0); // TfLiteTensor * , real_tensor
          std::cout << input_T->dims->data[0] << " " << input_T->dims->data[1]; 
          std::cout << " " << input_T->dims->data[2] << " " << input_T->dims->data[3] << std::endl;
          printf("\n\n=== Push image to input tensor (Before)===\n");
          // if(input_tensor == tmp){
          //   input_tensor+=0x305af00 / sizeof(float);
          // }
          // tmp = input_tensor;
          // Normalize code for pushing image to input tensor
          // TFLite's data tensor format :[B, H, W, C]
          // Opencv's data image  format :[W, H] 
          // PrintTensor(*input_T);
          printf("DEBUG_POINTER_ADDRESS : %p\n", (void*)input_tensor);
          printf("DEBUG_POINTER_VALUE : %f\n", *input_tensor);
          if(input_tensor == nullptr){
            printf("ERROR : get Nullptr!!!\n");
          }
          
          for (int w=0; w<width; w++){
            for (int h=0; h<height; h++){
              cv::Vec3b pixel = input[0].at<cv::Vec3b>(w, h);
              *(input_tensor + w * height*3 + h * 3) = ((float)pixel[0])/255.0;
              *(input_tensor + w * height*3 + h * 3 + 1) = ((float)pixel[1])/255.0;
              *(input_tensor + w * height*3 + h * 3 + 2) = ((float)pixel[2])/255.0;
            }
            // printf("DEBUG_POINTER %p\n", (void*)input_tensor);
          }
          
          printf("\n\n=== Push image to input tensor (After)===\n");
          // PrintTensor(*input_T);
          // for (int h=0; h<height; h++){
          //   for (int w=0; w<width; w++){
          //     cv::Vec3b pixel = input[0].at<cv::Vec3b>(w, h);
          //     *(input_tensor + h * width*3 + w * 3) = ((float)pixel[0])/255.0;
          //     *(input_tensor + h * width*3 + w * 3 + 1) = ((float)pixel[1])/255.0;
          //     *(input_tensor + h * width*3 + w * 3 + 2) = ((float)pixel[2])/255.0;
          //   }
          // }

          // Run inference
          uint64_t START = millis();
          printf("\n\n=== Interpreter Invoke (Before)===\n");
          TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
          uint64_t END = millis();
          uint64_t Invoke_time = END - START;
          printf("\n\n=== Interpreter Invoke (After)===\n");
          average_time += Invoke_time;

          #ifdef YOLO
          // Output parsing
          TfLiteTensor* cls_tensor = interpreter->output_tensor(1);
          TfLiteTensor* loc_tensor = interpreter->output_tensor(0);
          yolo_output_parsing(cls_tensor, loc_tensor);
          #endif

          // Output visualize
          #ifndef GPU
          yolo_output_visualize(image_name, image_number);
          #endif

          // Make txt file to get mAP
          // make_txt_to_get_mAP(yolo::YOLO_Parser::result_boxes, image_number, dot, N);

          // Re-initialize
          image_number+=1;
          input.clear();
          // interpreter->~Interpreter(); // Not use
        }
        //////////////////////////////////////////////////////////////////////////////////////////
        // Push result to time table
        printf("\033[0;31mDOT %d 's case's average invoke time [choose N=%d]: %0.2fms\033[0m\n", dot, N, float(average_time/IMG_set_num));
        time_table.push_back(float(average_time/IMG_set_num));
        if (dot == DOT -1){
          print_time_table(time_table);
        }
    }
    // Push each N's time table to parent time table
    DOT_table.push_back(time_table);
    time_table.clear();
  }
  /////////////
  // Search Best case recorded in DOT_table
  print_DOT_table(DOT_table);
  find_best_case(DOT_table);
  // cv::waitKey(0);
	// cv::destroyAllWindows();
  return 0;
}
