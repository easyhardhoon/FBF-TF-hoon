#include "tensorflow/lite/unit_handler.h"
#define SEQ 60000 //for input image
#define OUT_SEQ 1
#define catdog 

#define delegate_optimizing
#define Partition_Num 7// HOON --> number of all partition 14
#define Max_Delegated_Partitions_Num 1 // HOON -> max delegated partitions num 4

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


#ifdef catdog
void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(416,416)); //resize to 300x300
	input.push_back(cvimg_);
}
#endif

// extern std::vector<int> delegation_optimizer_v;
// std::vector<int> b_delegation_optimizer;  // <-> ?? 

int combination(int n, int r) {
    	if(n == r || r == 0) return 1; 
    	else return combination(n - 1, r - 1) + combination(n - 1, r);
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

	#ifdef catdog
	read_image_opencv("dog.jpg", input);
	std::cout << "Loading dog Image \n";
	#endif

	int max_delegated_partition_num = Max_Delegated_Partitions_Num;
	int test_number  = combination(Partition_Num, Max_Delegated_Partitions_Num);
	// printf("%d", test_number);
	// fix number use  {Partition_Num}  &  {Max_Delegated_Partitions_Num}


	#ifdef delegate_optimizing
	if(!bUseTwoModel){
		// 230406 TODO
		for (int loop_num=0; loop_num<test_number; loop_num++)
		{
			tflite::UnitHandler Uhandler(originalfilename);
			printf(".....................................................................................................\n");
			printf("%d loop starting.....\n", loop_num);
			if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input, loop_num, max_delegated_partition_num, test_number) != kTfLiteOk)
			{
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
			}
			// std::cout << "-----------------------------------TODO: " <<  b_delegation_optimizer.size() << std::endl;
			printf("%d loop End.....\n", loop_num);
			// if (loop_num >= 16382)  printf("\033[0;31m choose 14 delegation node\033[0m\n");
			// else if (loop_num >= 16368)  printf("\033[0;31m choose 13 delegation node\033[0m\n");
			// else if (loop_num >= 16277) printf("\033[0;31m choose 12 delegation node\033[0m\n");
			// else if (loop_num >= 15913)  printf("\033[0;31m choose 11 delegation node\033[0m\n");
			// else if (loop_num >= 14912) printf("\033[0;31m choose 10 delegation node\033[0m\n");
			// else if (loop_num >= 12910) printf("\033[0;31m choose 9 delegation node\033[0m\n");
			// else if (loop_num >= 9907) printf("\033[0;31m choose 8 delegation node\033[0m\n");
			// else if (loop_num >= 6475) printf("\033[0;31m choose 7 delegation node\033[0m\n");
			// else if (loop_num >= 3472) printf("\033[0;31m choose 6 delegation node\033[0m\n");
			// else if (loop_num >= 1470) printf("\033[0;31m choose 5 delegation node\033[0m\n");
			// else if (loop_num >= 469) printf("\033[0;31m choose 4 delegation node\033[0m\n");
			// else if (loop_num >= 105) printf("\033[0;31m choose 3 delegation node\033[0m\n");
			// else if (loop_num >= 14)  printf("\033[0;31m choose 2 delegation node \033[0m\n");
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
