#include <yolo_parser.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
using namespace std;
yolo::YOLO_Parser yolo_parser;
std::vector<yolo::YOLO_Parser::BoundingBox> yolo::YOLO_Parser::result_boxes;
std::vector<std::vector<float>> yolo::YOLO_Parser::real_bbox_cls_vector; 
std::vector<int> yolo::YOLO_Parser::real_bbox_cls_index_vector;
std::vector<std::vector<int>> yolo::YOLO_Parser::real_bbox_loc_vector;

uint64_t millis() {
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    return ms; 
};

std::map<std::string, std::pair<int,int>> input_map = {
  {"yolo", {416,416}},
  {"lane", {512,256}},
  {"move", {192,192}}
};

void yolo_output_parsing(TfLiteTensor* cls_tensor, TfLiteTensor* loc_tensor){
  printf("\033[0;33mStart YOLO parsing\033[0m\n");
  std::vector<int> real_bbox_index_vector;
  real_bbox_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_cls_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_cls_vector.clear();
  yolo::YOLO_Parser::real_bbox_loc_vector.clear();
  yolo::YOLO_Parser::result_boxes.clear();
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
};

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

void yolo_output_visualize(std::string image_name, int image_number){
  std::vector<yolo::YOLO_Parser::BoundingBox> bboxes = yolo::YOLO_Parser::result_boxes;
	std::string window_name = std::to_string(image_number) + "'s  parsed image";
	// cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
	cv::resize(image, image, cv::Size(416,416)); 
  	if (!image.empty()) {
  	    yolo_parser.visualize_with_labels(image, bboxes, labelDict);
    		// cv::imshow(window_name, image);
	}
};

void make_txt_to_get_mAP(std::vector<yolo::YOLO_Parser::BoundingBox>& result_boxes, int image_number, int dot_number, int choose_num){
  std::string foldername =  "../../mAP_TF/input/DOT/choose_" + std::to_string(choose_num) + "/DOT_" + std::to_string(dot_number) + "/";
	std::string filename = std::to_string(image_number) + ".txt";
  std::string dst = foldername + filename;
	std::ofstream outFile(dst);
	if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << dst << std::endl;
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
};


void PrintTensor(TfLiteTensor& tensor){
    std::cout << "[Print Tensor]" << "\n";
  int tensor_channel_idx = tensor.dims->size-1;
  int tensor_data_ch_size = tensor.dims->data[tensor_channel_idx];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< tensor.dims->size; i++){
    if(i == 2){
      tensor_axis = tensor.dims->data[i];
    }
    tensor_data_size *= tensor.dims->data[i]; 
  }
  std::cout << " Number of data : " << tensor_data_size << "\n";
  std::cout << " Tensor DATA " << "\n";
  if(tensor.type == TfLiteType::kTfLiteFloat32){
    std::cout << "[FLOAT32 TENSOR]" << "\n";
    auto data_st = (float*)tensor.data.data;
    for(int i=0; i<tensor_data_ch_size; i++){
      std::cout << "CH [" << i << "] \n";
      for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
        float data = *(data_st+(i+j*tensor_data_ch_size));
        if (data == 0) {
          printf("%0.6f ", data);
        }
        else if (data != 0) {
            printf("%s%0.6f%s ", C_GREN, data, C_NRML);
        }
        if (j % tensor_axis == tensor_axis-1) {
          printf("\n");
        }
      }
      std::cout << "\n";
    }
  }
};

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
};
void printCombination(int n, int k, int kth) {
    std::vector<int> result;
    mother_vec = make_mother_vec(-1, result, k, n);
    result = mother_vec[kth];
    std::cout <<"[";
    for (auto &data : result) std::cout << data << " ";
    std::cout <<"]";
    mother_vec.clear();
    return;
};
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
};

int combination(int n, int r) {
    	if(n == r || r == 0) return 1; 
    	else return combination(n - 1, r - 1) + combination(n - 1, r);
};

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
};