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

#define YOLO_INPUT "../../mAP_TF/input/images-optional/"
#define Partition_Num 7  // nCr --> "n"  // for YOLOv4-tiny
// #define Max_Delegated_Partitions_Num 1  // nCr --> "r"  // hyper-param // Not use in full-auto
#define GPU
#define IMG_set_num 100 // "300" for mAP , "100" for DOT

std::vector<float> time_table;
std::vector<std::vector<float>> DOT_table;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

uint64_t millis() {
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    return ms; 
}
std::vector<std::vector<int>> mother_vec;
std::vector<std::vector<int>>make_mother_vec(int start , std::vector<int> vec, int n, int total) {
    if (vec.size() == n) {
    		mother_vec.push_back(vec);
    		return mother_vec;
    	}
    	for (int i = start + 1; i < total; i++) {
    		vec.push_back(i);
    		make_mother_vec(i, vec, n, total);
    		vec.pop_back();
    	}
    	return mother_vec; 
}
void printCombination(int n, int k, int kth) {
    std::vector<int> result;
    mother_vec = make_mother_vec(-1, result, k, n);
    result = mother_vec[kth];
    std::cout <<"[";
    for (auto &data : result) std::cout << data << " ";
    std::cout <<"]";
    mother_vec.clear();
    return;
}
void print_time_table(std::vector<float> time_table){
  std::cout << "\033[0;31mLatency for each case in DOT\033[0m : " <<std::endl;
  double min = *min_element(time_table.begin(), time_table.end());
  float bias = 0.3;
  for (int i=0;i< time_table.size(); i++){
      if(time_table.at(i) < min + bias) { 
          printf("\033[0;31m%d case's latency is %0.2fms\033[0m\n",i,time_table.at(i));
      }
      else{
          std::cout << i << " case's latency is : " << time_table.at(i) << "ms" << std::endl;
      }
  }
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
}
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
	
	//For Debugging
	N=100;

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
	tflite::PrintInterpreterState(interpreter.get());  //For debugging model info
        //////////////////////////////////////////////////////////////////////////////////////////
        // Push test image to input_tensor
        for (int loop_num=0;loop_num<IMG_set_num;loop_num++){ 
          // Load image 
          std::string image_name = YOLO_INPUT + std::to_string(image_number) + ".jpg";
          read_image_opencv(image_name, input);

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
          uint64_t START = millis();
          //TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
          interpreter->Invoke(); // ISSUE
          uint64_t END = millis();
          uint64_t Invoke_time = END - START;
          printf("\n\n=== Interpreter Invoke ===\n");
          average_time += Invoke_time;

          // Output parsing
          TfLiteTensor* cls_tensor = interpreter->output_tensor(1);
          TfLiteTensor* loc_tensor = interpreter->output_tensor(0);
          yolo_output_parsing(cls_tensor, loc_tensor);

          // Output visualize
          #ifndef GPU
          yolo_output_visualize(image_name, image_number);
          #endif

          // Make txt file to get mAP
          // make_txt_to_get_mAP(yolo::YOLO_Parser::result_boxes, image_number, dot, Max_Delegated_Partitions_Num);

          // Re-initialize
          image_number+=1;
          input.clear();
          // interpreter->~Interpreter(); // Not use
        }
        //////////////////////////////////////////////////////////////////////////////////////////
        // Push result to time table
        // printf("\033[0;31mDOT %d 's case's average invoke time [choose N=%d]: %0.2fms\033[0m\n", dot, float(average_time/IMG_set_num), N);
        time_table.push_back(float(average_time/IMG_set_num));
        if (dot == DOT -1){
          // print_time_table(time_table);
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
  cv::waitKey(0);
	cv::destroyAllWindows();
  return 0;
}
