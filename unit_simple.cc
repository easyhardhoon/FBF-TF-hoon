#include "tensorflow/lite/unit_handler.h"
#include "tensorflow/lite/core/subgraph.h"
// #include "tensorflow/lite/interpreter.h"
#define SEQ 60000 //for input image
#define OUT_SEQ 1

#define yolo //    Y / N
#define delegate_optimizing

#ifdef yolo
#define Partition_Num 7 // HOON 
#define Max_Delegated_Partitions_Num 1 // HOON 
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


#ifdef yolo
void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
	cv::Mat cvimg_;
	// cv::resize(cvimg, cvimg_, cv::Size(320,320)); //resize to 300x300   // 416 * 416 --> original image size
	cv::resize(cvimg, cvimg_, cv::Size(416,416)); //resize to 300x300   // 416 * 416 --> original image size
	// cvimg_.convertTo(cvimg_, CV_32F, 1.0 / 255.0);
	input.push_back(cvimg_);
}
#endif

// extern std::vector<int> delegation_optimizer_v;
// std::vector<int> b_delegation_optimizer;  // <-> ?? 

int combination(int n, int r) {
    	if(n == r || r == 0) return 1; 
    	else return combination(n - 1, r - 1) + combination(n - 1, r);
}


void output_data_postprocessing(const std::vector<int>& real_bbox_cls_index_vector, \
  const std::vector<std::vector<float>>& real_bbox_cls_vector, const std::vector<std::vector<float>>& real_bbox_loc_vector, int fnum)
  {
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
        {52, "hot dog"},   {53, "pizza"},   {54, "donut"},       {55, "cake"},
        {56, "chair"},     {57, "sofa"},    {58, "potted_plant"}, {59, "bed"},
        {60, "dining_table"}, {61, "toilet"}, {62, "tvmonitor"}, {63, "laptop"},
        {64, "mouse"},     {65, "remote"},  {66, "keyboard"},    {67, "cell_phone"},
        {68, "microwave"}, {69, "oven"},    {70, "toaster"},     {71, "sink"},
        {72, "refrigerator"}, {73, "book"}, {74, "clock"},       {75, "vase"},
        {76, "scissors"},  {77, "teddy_bear"}, {78, "hair_drier"}, {79, "toothbrush"}
    };
	int n=0;
	std::string filename = "../mAP_TF/input/detection-results/" + std::to_string(fnum) + ".txt";
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
	for (auto i : real_bbox_cls_index_vector) { 
		auto object_name = labelDict[i];
		auto cls_data = real_bbox_cls_vector[n][i];
		auto loc_data = real_bbox_loc_vector[n];
		outFile << object_name << " " << cls_data << " ";
		for (auto j : loc_data){
			outFile << j << " ";
		}
		outFile << std::endl;
		n+=1;
	}
	outFile.close();
	// std::cout << real_bbox_cls_index_vector.size() << std::endl;
	// std::cout << real_bbox_cls_index_vector.size() << std::endl;
	// std::cout << real_bbox_cls_index_vector.size() << std::endl;
	tflite::Subgraph::real_bbox_cls_index_vector.clear();
	tflite::Subgraph::real_bbox_cls_vector.clear();
	tflite::Subgraph::real_bbox_loc_vector.clear();
	// std::cout << real_bbox_cls_index_vector.size() << std::endl;
	// std::cout << real_bbox_cls_index_vector.size() << std::endl;
	// std::cout << real_bbox_cls_index_vector.size() << std::endl;
  }

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
	// read_image_opencv("data/dog-group.jpg", input);
	// read_image_opencv("data/dog_horse_person.jpg", input);
	// read_image_opencv("data/dog.jpg", input);
	// read_image_opencv("../mAP_TF/input/images-optional/2007_000027.jpg", input);
	// read_image_opencv("data/cats.jpg", input);
	// std::cout << "Loading dog Image \n";
	#endif

	int max_delegated_partition_num = Max_Delegated_Partitions_Num;
	int test_number  = combination(Partition_Num, Max_Delegated_Partitions_Num);
	// printf("%d", test_number);
	// fix number use  {Partition_Num}  &  {Max_Delegated_Partitions_Num}

	#ifdef delegate_optimizing
	if(!bUseTwoModel){
		test_number = 194;  // HOONING : Debugging for CPU YOLO-output parsing
		// test_number = 2;  // HOONING : Debugging for CPU YOLO-output parsing
		int fnum = 0;
		for (int loop_num=0; loop_num<test_number; loop_num++)
		{
			// int fnum = 1 + loop_num;
		    fnum+=1;
			input.clear();  // NOTE : should clear "input" vector
			std::string filename = "../mAP_TF/input/images-optional/" + std::to_string(fnum) + ".jpg";
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
		    // --------------------------------------------------------------
			std::vector<int> a = tflite::Subgraph::real_bbox_cls_index_vector;
			std::vector<std::vector<float>> b = tflite::Subgraph::real_bbox_cls_vector;
			std::vector<std::vector<float>> c = tflite::Subgraph::real_bbox_loc_vector;
		    output_data_postprocessing(tflite::Subgraph::real_bbox_cls_index_vector, tflite::Subgraph::real_bbox_cls_vector, tflite::Subgraph::real_bbox_loc_vector, fnum);
			// --------------------------------------------------------------
		}
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
