#include "tensorflow/lite/unit_handler.h"
#include "tensorflow/lite/core/subgraph.h"
#include "minimal_yolo.h"

#define Partition_Num 7 // HOON 
#define Max_Delegated_Partitions_Num 7 // HOON 
#define CPU
using namespace cv;
using namespace std;

// For Delegation Optimizing
int combination(int n, int r) {
    	if(n == r || r == 0) return 1; 
    	else return combination(n - 1, r - 1) + combination(n - 1, r);
}

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

int main(int argc, char* argv[])
{
	const char* originalfilename = argv[1];
	if (argc != 2) {
    	fprintf(stderr, "minimal <tflite model>\n");
    	return 1;
  	}
	vector<cv::Mat> input;

	int max_delegated_partition_num = Max_Delegated_Partitions_Num;
	int test_number  = combination(Partition_Num, Max_Delegated_Partitions_Num);
	test_number = 1;  // FULL 330
	int fnum = 0;	    // FULL	
	for (int loop_num=0; loop_num<test_number; loop_num++)
	{
	    fnum+=1;
		input.clear();  
		std::string filename = "../../mAP_TF/input/images-optional/" + std::to_string(fnum) + ".jpg";
		read_image_opencv(filename, input);

        ////////////////////////////////////////////////////////////////////////////////////////////
		tflite::UnitHandler Uhandler(originalfilename);
		printf("%d loop starting.....\n", loop_num);
		if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input, loop_num, max_delegated_partition_num, test_number) != kTfLiteOk){
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
		}
		printf("%d loop End.....\n", loop_num);
  		////////////////////////////////////////////////////////////////////////////////////////////
		
		std::cout << "1111 " <<std::endl;

		// Output parsing
		#ifdef CPU
		// tflite::UnitCPU Unit;
		TfLiteTensor* cls_tensor = tflite::UnitCPU::UnitCPU::GetInterpreter()->output_tensor(1);
  		TfLiteTensor* loc_tensor = Unit.GetInterpreter()->output_tensor(0);
		#endif
		std::cout << "2222 " <<std::endl;

		#ifdef GPU
		TfLiteTensor* cls_tensor = tflite::UnitGPU->GetInterpreter()->output_tensor(1);
  		TfLiteTensor* loc_tensor = tflite::UnitGPU->GetInterpreter()->output_tensor(0);
		#endif
  		yolo_output_parsing(cls_tensor, loc_tensor);
		std::cout << "3333 " <<std::endl;

  		// Output visualize
  		yolo_output_visualize(filename);
		std::cout << "4444 " <<std::endl;

		// Make txt file to get mAP
		make_txt_to_get_mAP(yolo::YOLO_Parser::result_boxes, fnum);
		
		std::cout << "5555 " <<std::endl;
	}


	
}
