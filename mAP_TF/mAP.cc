#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

std::vector<int> get_cls_index(std::vector<std::vector<float>> real_bbox_cls_vector){
  std::vector<int> real_bbox_cls_index_vector;
  float max=0;
  int max_index = -1;
  int index = 0;
  for (auto i : real_bbox_cls_vector) { 
    index = 0;
		for (auto j : i) { 
      if (j > max){
        max = j;
        max_index = index;
      }
      index+=1;
		}
    real_bbox_cls_index_vector.push_back(max_index);
		// std::cout << std::endl << std::endl;
	}
  printf("real_bbox_cls_index vector : ");
  for (auto i :real_bbox_cls_index_vector)
    printf("\033[0;31m%d\033[0m ", i);
  printf("\n\n");
  return real_bbox_cls_index_vector;
}

std::vector<std::vector<float>> readDataFromFile(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream inFile(filename);
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            std::vector<float> row;
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                row.push_back(value);
            }
            data.push_back(row);
        }
        inFile.close();
    }
    return data;
}

int main(int argc, char* argv[]) {
    std::vector<std::vector<float>> real_bbox_cls_vector = readDataFromFile("./cls.txt");
    std::vector<std::vector<float>> real_bbox_loc_vector = readDataFromFile("./loc.txt");
    std::vector<int> real_bbox_cls_index_vector = get_cls_index(real_bbox_cls_vector);
    printf("size : %d\n", real_bbox_cls_vector.size());
    printf("size : %d\n", real_bbox_loc_vector.size());
    printf("size : %d\n", real_bbox_cls_index_vector.size());
    return 0;
}



