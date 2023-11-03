#include <iostream>
#include <vector>
using namespace std;

std::vector<std::vector<int>> mother_vector;

std::vector<std::vector<int>> make_mother_vector(int start, std::vector<int> &input_vector, std::vector<int> &vector, int n) {
    if (vector.size() == n) {
        mother_vector.push_back(vector);
        return mother_vector;
    }
    for (int i = start + 1; i < input_vector.size(); i++) {
        vector.push_back(input_vector[i]);
        make_mother_vector(i, input_vector, vector, n);
        vector.pop_back();
    }
    return mother_vector;
}

int main() {
    //std::vector<int> input_vector = {5, 9, 13, 17, 20, 24, 28, 32, 39, 43, 47, 51, 55, 62, 66, 70, 74, 78, 85, 89, 93, 97, 101, 105, 109};
    std::vector<int> input_vector = {5, 9, 13, 17, 20, 24, 28,32,39,43,47,51};
    for (int n = 1; n <= input_vector.size(); n++) {
        std::vector<int> baby_vector;
        mother_vector = make_mother_vector(-1, input_vector, baby_vector, n);
    }
    int total_num=0;
    for (const auto &vec : mother_vector) {
	 // Each 1D vector [Ex {5,9,13}]. Each case is saved in mother_vector[i] 
         for (int i : vec) {
             std::cout << i << " ";
	     //total_num+=1;
         }
         std::cout << std::endl;
    }
    //std::cout << "Total case num : " << total_num << std::endl;
    return 0;
}

