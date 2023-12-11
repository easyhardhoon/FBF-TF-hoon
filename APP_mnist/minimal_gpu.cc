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
#include <typeinfo>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <cstdarg>
#include <queue>
#include <ctime>
#include <functional>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <chrono>



#define mnist
#define SEQ 60000
#define OUT_SEQ 1

using namespace cv;
using namespace std;

#ifdef mnist
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(const char * filename, vector<cv::Mat>& vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
    file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < SEQ; ++i){
			cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r){
				for (int c = 0; c < n_cols; ++c){
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
      // cout << "Get " << i << " Images" << "\n";
		}
	}
	else {
		cout << "file open failed" << endl;
	}
}

void read_Mnist_Label(const char * filename, vector<unsigned char> &arr) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < SEQ; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				// cout << (int)temp << " ";
				arr.push_back((unsigned char)temp);
			}
		}
	}
	else {
        cout << "file open failed" << endl;
    }
}
#endif


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

uint64_t millis()
{
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    return ms; 
}

uint64_t micros()
{
    uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(\
            std::chrono::high_resolution_clock::now().time_since_epoch())\
            .count();
    return us; 
}
uint64_t nanos()
{
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(\
            std::chrono::high_resolution_clock::now().time_since_epoch())\
            .count();
    return ns; 
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* filename1 = argv[2];
  const char* filename2 = argv[3];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
  tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  
  // --------------------------------------------------------------------------------------------
  // NOTE gpu delegate. interpreter ( CPU --> GPU )
  TfLiteDelegate *MyDelegate = NULL;

  // const TfLiteGpuDelegateOptionsV2 options = {
      // .is_precision_loss_allowed = 1, 
      // .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
      // .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
      // .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      // .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      // .experimental_flags=4,
      // .max_delegated_partitions=5,
  // };
  const TfLiteGpuDelegateOptionsV2 options = {
      .is_precision_loss_allowed = 0, 
      .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
      .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
      .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .experimental_flags = 1,
      .max_delegated_partitions = 1,
  };
  MyDelegate = TfLiteGpuDelegateV2Create(&options);
  if(interpreter->ModifyGraphWithDelegate(MyDelegate) != kTfLiteOk) {
      printf("Unable to Use GPU Delegate\n");
      return 0;
  }
  // --------------------------------------------------------------------------------------------

  // Allocate tensor buffers.
  printf("=====AllocateTensors()=====\n\n");
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  tflite::PrintInterpreterState(interpreter.get());


  printf("\n=====set_input======\n\n");

  vector<cv::Mat> input;
  vector<unsigned char> arr;
  #ifdef mnist
  std::cout << "Loading images \n";
  read_Mnist(filename1, input);
  std::cout << "Loading Labels \n";
  read_Mnist_Label(filename2, arr);
  std::cout << "Loading Mnist Image, Label Complete \n";
  #endif

  float average_time = 0;
  float average_accuarcy = 0;
  for(int k=0;k<SEQ;k++)
  {
    for (int i=0; i<28; i++)
    {
      for (int j=0; j<28; j++)
        {
        interpreter->typed_input_tensor<float>(0)[i*28 + j] = ((float)input[k].at<uchar>(i, j)/255.0); 
        if(interpreter->typed_input_tensor<float>(0)[i*28 + j] != 0)
	      {
          printf("\033[0;31m%0.4f\033[0m",interpreter->typed_input_tensor<float>(0)[i*28 + j]); 
        }  
        else
        {
        printf("%0.4f",interpreter->typed_input_tensor<float>(0)[i*28 + j]);
        }
      }
      printf("\n");
    }
    printf("\n=====START Invoke=====\n\n");
    uint64_t START = nanos();
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    uint64_t END = nanos();
    uint64_t Invoke_time = END - START;
    printf("\n=====End Invoke=====\n\n");
    printf("single data's invoke time is %0.6f ms\n", (float)Invoke_time / (float)1000000);
    printf("=====get_output=====\n\n");
    float max =0;
    for (int n=0;n<10;n++)
    {
      if (interpreter->typed_output_tensor<float>(0)[n] > max)
        max = interpreter->typed_output_tensor<float>(0)[n];
      printf("%d's data's output[label:%d] : %f\n", k, n,interpreter->typed_output_tensor<float>(0)[n]);
    }
    average_accuarcy += max;
    printf("single data's accuarcy is %f \n", max);
    average_time +=  (float)Invoke_time / (float)1000000;

    printf("\n%d'sDATA\n", k+1);
  }
  printf(" >>>>>>>>>>>> model's average accuracy : %.6f %\n", average_accuarcy / (float)SEQ * 100);
  printf(" >>>>>>>>>>>> model's average invoke time : %.6f ms\n", average_time / (float)SEQ);
  return 0;
}
