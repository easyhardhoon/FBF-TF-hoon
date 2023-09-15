#include "tensorflow/lite/unit_handler.h"
#include "tensorflow/lite/core/subgraph.h"
#define SEQ 60000 //for input image
#define OUT_SEQ 1

#define yolo //    Y / N
#define delegate_optimizing

#ifdef yolo
#define Partition_Num 7 // HOON 
#define Max_Delegated_Partitions_Num 7 // HOON 
#endif

#ifndef yolo
#define mnist
#define Partition_Num 14// HOON 
#define Max_Delegated_Partitions_Num 1 // HOON 
#endif

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

void read_Mnist(string filename, vector<cv::Mat>& vec) {
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
            cout << "Get " << i << " Images" << "\n";
		}
	}
	else {
		cout << "file open failed" << endl;
	}
}

void read_Mnist_Label(string filename, vector<unsigned char> &arr) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < SEQ; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				cout << (int)temp << " ";
				arr.push_back((unsigned char)temp);
			}
		}
	}
	else {
        cout << "file open failed" << endl;
    }
}
#endif

// For Delegation Optimizing
int combination(int n, int r) {
    	if(n == r || r == 0) return 1; 
    	else return combination(n - 1, r - 1) + combination(n - 1, r);
}

#ifdef yolo
void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB); 
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(416,416));
	input.push_back(cvimg_);

}
#endif

// void YOLO_parsing(std::vector<tflite::YOLO_Parser::BoundingBox>& result_boxes, int fnum, std::map<int, std::string>& labelDict)
//   {
// 	std::string filename = "../mAP_TF/input/detection-results/" + std::to_string(fnum) + ".txt";
// 	std::ofstream outFile(filename);
// 	if (!outFile.is_open()) {
//         std::cerr << "Error: Unable to open file " << filename << std::endl;
//         return;
//     }
// 	for (int i=0; i <result_boxes.size(); i++) { 
// 		auto object_name = labelDict[result_boxes[i].class_id];
// 		auto left = result_boxes[i].left;
// 		auto top = result_boxes[i].top;
// 		auto right = result_boxes[i].right;
// 		auto bottom = result_boxes[i].bottom;
// 		auto cls_data = result_boxes[i].score;
// 		outFile << object_name << " " <<  cls_data << " ";
// 		outFile << left << " " << top << " " << right << " " << bottom; 
// 		outFile << std::endl;
// 	}
// 	outFile.close();
//   }

// void visualize_with_labels(cv::Mat& image, const std::vector<tflite::YOLO_Parser::BoundingBox>& bboxes, std::map<int, std::string>& labelDict) {
//     for (const tflite::YOLO_Parser::BoundingBox& bbox : bboxes) {
//         int x1 = bbox.left;
//         int y1 = bbox.top;
//         int x2 = bbox.right;
//         int y2 = bbox.bottom;
//         cv::RNG rng(bbox.class_id);
//         cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//         int label_x = x1;
//         int label_y = y1 - 20;

//         cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
//         std::string object_name = labelDict[bbox.class_id];
//         float confidence_score = bbox.score;
//         std::string label = object_name + ": " + std::to_string(confidence_score);
//         cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
//         cv::rectangle(image, cv::Point(x1, label_y - text_size.height), cv::Point(x1 + text_size.width, label_y + 5), color, -1);
//         cv::putText(image, label, cv::Point(x1, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
//     }
// }

int main(int argc, char* argv[])
{
	const char* originalfilename;
	const char* quantizedfilename;
	bool bUseTwoModel = false;
	if (argc == 2) {
		std::cout << "Got One Model \n";
		originalfilename = argv[1];
	}
	else if(argc > 2){
		std::cout << "Got Two Model \n";
		bUseTwoModel = true;
		originalfilename = argv[1];
		quantizedfilename = argv[2];
	}
	else{
			fprintf(stderr, "minimal <tflite model>\n");
			return 1;
	}
	vector<cv::Mat> input;
	vector<unsigned char> arr;
	// vector<int> delegation_optimizer_v;
	#ifdef mnist
	std::cout << "Loading images \n";
        read_Mnist("train-images-idx3-ubyte", input);
	std::cout << "Loading Labels \n";
	read_Mnist_Label("train-labels-idx1-ubyte", arr);
	std::cout << "Loading Mnist Image, Label Complete \n";
	#endif

	#ifdef yolo
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
	#endif

	#ifdef delegate_optimizing
	int max_delegated_partition_num = Max_Delegated_Partitions_Num;
	int test_number  = combination(Partition_Num, Max_Delegated_Partitions_Num);
	if(!bUseTwoModel){
		// test_number = 1;  // FULL 330
		int fnum = 0;	    // FULL	
		for (int loop_num=0; loop_num<test_number; loop_num++)
		{
		    fnum+=1;
			input.clear();  
			std::string filename = "../../mAP_TF/input/images-optional/" + std::to_string(fnum) + ".jpg";
			read_image_opencv(filename, input);
			std::cout << filename << std::endl;
			tflite::UnitHandler Uhandler(originalfilename);
			printf(".....................................................................................................\n");
			printf("%d loop starting.....\n", loop_num);
			if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input, loop_num, max_delegated_partition_num, test_number) != kTfLiteOk)
			{
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
			}
			printf("%d loop End.....\n", loop_num);
  			////////////////////////////////////////////////////////////////////////////////////////////
			// std::vector<tflite::YOLO_Parser::BoundingBox> bboxes = tflite::YOLO_Parser::result_boxes;
			// YOLO_parsing(bboxes, fnum, labelDict);
			// // visualize
			// std::string window_name = std::to_string(fnum) + "'s parsed image";
			// cv::namedWindow(window_name, cv::WINDOW_NORMAL);
			// cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
			// cv::resize(image, image, cv::Size(416,416)); 
    		// if (!image.empty()) {
    		//     visualize_with_labels(image, bboxes, labelDict);
        	// 	cv::imshow(window_name, image);
			// }
			// else {
    		// 	std::cerr << "Error: Unable to load the image: " << filename << std::endl;
			// }
    	}
		// cv::waitKey(0);
		// cv::destroyAllWindows();
		////////////////////////////////////////////////////////////////////////////////////////////

	}

	else{
		int loop_num = 0;
		tflite::UnitHandler Uhandler(originalfilename, quantizedfilename);
		if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input, loop_num, max_delegated_partition_num, test_number) != kTfLiteOk){
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
		}
	}
	#endif
	#ifndef delegate_optimizing
	if(!bUseTwoModel){
		tflite::UnitHandler Uhandler(originalfilename);
		// HOON ==> make extra loop for deleation optimizing???
		// 230406 TODO
		if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input, loop_num) != kTfLiteOk){
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
		}		
	}
	
	else{
		tflite::UnitHandler Uhandler(originalfilename, quantizedfilename);
		if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input, loop_num) != kTfLiteOk){
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
		}
	}
	#endif


	
}
