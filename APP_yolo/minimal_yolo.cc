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
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "yolo_parser.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>
using namespace std;

#define MNIST_INPUT "/home/pi/EAI_TfLite/01_minimal/mnist_images"
#define MNIST_LABEL "/home/pi/EAI_TfLite/01_minimal/mnist_labels"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, cv::COLOR_BGR2RGB); 
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(416,416));
	input.push_back(cvimg_);

}

void YOLO_parsing(std::vector<yolo::YOLO_Parser::BoundingBox>& result_boxes, int fnum, std::map<int, std::string>& labelDict)
  {
	std::string output_filename = "../../mAP_TF/input/detection-results/" + std::to_string(fnum) + ".txt";
	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << output_filename << std::endl;
        return;
    }
	for (int i=0; i <result_boxes.size(); i++) { 
		auto object_name = labelDict[result_boxes[i].class_id];
		auto left = result_boxes[i].left;
		auto top = result_boxes[i].top;
		auto right = result_boxes[i].right;
		auto bottom = result_boxes[i].bottom;
		auto cls_data = result_boxes[i].score;
		outFile << object_name << " " <<  cls_data << " ";
		outFile << left << " " << top << " " << right << " " << bottom; 
		outFile << std::endl;
	}
	outFile.close();
  }


void visualize_with_labels(cv::Mat& image, const std::vector<yolo::YOLO_Parser::BoundingBox>& bboxes, std::map<int, std::string>& labelDict) {
    for (const yolo::YOLO_Parser::BoundingBox& bbox : bboxes) {
        int x1 = bbox.left;
        int y1 = bbox.top;
        int x2 = bbox.right;
        int y2 = bbox.bottom;
        cv::RNG rng(bbox.class_id);
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        int label_x = x1;
        int label_y = y1 - 20;

        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
        std::string object_name = labelDict[bbox.class_id];
        float confidence_score = bbox.score;
        std::string label = object_name + ": " + std::to_string(confidence_score);
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::rectangle(image, cv::Point(x1, label_y - text_size.height), cv::Point(x1 + text_size.width, label_y + 5), color, -1);
        cv::putText(image, label, cv::Point(x1, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

// Should initialize out of main func
std::vector<yolo::YOLO_Parser::BoundingBox> yolo::YOLO_Parser::result_boxes;
std::vector<std::vector<float>> yolo::YOLO_Parser::real_bbox_cls_vector; 
std::vector<int> yolo::YOLO_Parser::real_bbox_cls_index_vector;
std::vector<std::vector<int>> yolo::YOLO_Parser::real_bbox_loc_vector;


int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load mnist input images
  vector<cv::Mat> input;
  int fnum =1;
	std::map<int, std::string> labelDict = {
        {0, "person"},     {1, "bicycle"},   {2, "car"},          {3, "motorbike"},
        {4, "aeroplane"},  {5, "bus"},       {6, "train"},        {7, "truck"},
        {8, "boat"},       {9, "traffic_light"}, {10, "fire_hydrant"}, {11, "stop_sign"},
        {12, "parking_meter"}, {13, "bench"}, {14, "bird"},       {15, "cat"},
        {16, "dog"},       {17, "horse"},    {18, "sheep"},       {19, "cow"},
        {20, "elephant"},  {21, "bear"},     {22, "zebra"},       {23, "giraffe"},
        {24, "backpack"},  {25, "umbrella"}, {26, "handbag"},     {27, "tie"},
        {28, "suitcase"},  {29, "frisbee"},  {30, "skis"},        {31, "snowboard"},
        {32, "sports_ball"}, {33, "kite"},   {34, "baseball_bat"}, {35, "baseball_glove"},
        {36, "skateboard"}, {37, "surfboard"}, {38, "tennis_racket"}, {39, "bottle"},
        {40, "wine_glass"}, {41, "cup"},     {42, "fork"},        {43, "knife"},
        {44, "spoon"},     {45, "bowl"},    {46, "banana"},      {47, "apple"},
        {48, "sandwich"},  {49, "orange"},  {50, "broccoli"},    {51, "carrot"},
        {52, "hot_dog"},   {53, "pizza"},   {54, "donut"},       {55, "cake"},
        {56, "chair"},     {57, "sofa"},    {58, "potted_plant"}, {59, "bed"},
        {60, "dining_table"}, {61, "toilet"}, {62, "tvmonitor"}, {63, "laptop"},
        {64, "mouse"},     {65, "remote"},  {66, "keyboard"},    {67, "cell_phone"},
        {68, "microwave"}, {69, "oven"},    {70, "toaster"},     {71, "sink"},
        {72, "refrigerator"}, {73, "book"}, {74, "clock"},       {75, "vase"},
        {76, "scissors"},  {77, "teddy_bear"}, {78, "hair_drier"}, {79, "toothbrush"}
    };
  std::string image_name = "../../mAP_TF/input/images-optional/" + std::to_string(fnum) + ".jpg";
  read_image_opencv(image_name, input);

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
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
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`


  // auto output_tensor = interpreter->typed_output_tensor<float>(0);

  
  // const std::vector<TfLiteTensor*>& output_tensors = interpreter->outputs();
  // TfLiteTensor* cls_tensor = output_tensors[0];  
  // TfLiteTensor* loc_tensor = output_tensors[1];
  //----------------------------------------------------------------
  //----------------------------------------------------------------

  yolo::YOLO_Parser yolo_parser;
  printf("\033[0;33mStart YOLO parsing\033[0m\n");
  std::vector<int> real_bbox_index_vector;
  real_bbox_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_cls_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_cls_vector.clear();
  yolo::YOLO_Parser::real_bbox_loc_vector.clear();
  yolo::YOLO_Parser::result_boxes.clear();
  ///////////////////////////////////////////////////
  TfLiteTensor* cls_tensor = interpreter->output_tensor(212);
  TfLiteTensor* loc_tensor = interpreter->output_tensor(233);
  ///////////////////////////////////////////////////
  yolo_parser.make_real_bbox_cls_vector(cls_tensor, real_bbox_index_vector,
                                         yolo::YOLO_Parser::real_bbox_cls_vector);
  yolo::YOLO_Parser::real_bbox_cls_index_vector = \
              yolo_parser.get_cls_index(yolo::YOLO_Parser::real_bbox_cls_vector); 
  yolo_parser.make_real_bbox_loc_vector(loc_tensor, real_bbox_index_vector, 
                                        yolo::YOLO_Parser::real_bbox_loc_vector);
  float iou_threshold = 0.5;
  yolo_parser.PerformNMSUsingResults(real_bbox_index_vector, yolo::YOLO_Parser::real_bbox_cls_vector, 
        yolo::YOLO_Parser::real_bbox_loc_vector, iou_threshold,yolo::YOLO_Parser::real_bbox_cls_index_vector);
  printf("\033[0;33mEND YOLO parsing\033[0m\n");



  // @@@
  std::vector<yolo::YOLO_Parser::BoundingBox> bboxes = yolo::YOLO_Parser::result_boxes;
	YOLO_parsing(bboxes, fnum, labelDict);
	// visualize
	std::string window_name = std::to_string(fnum) + "'s parsed image";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
	cv::resize(image, image, cv::Size(416,416)); 
  	if (!image.empty()) {
  	    visualize_with_labels(image, bboxes, labelDict);
    		cv::imshow(window_name, image);
	}
  cv::waitKey(0);
	cv::destroyAllWindows();
  return 0;
}
