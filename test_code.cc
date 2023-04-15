#include <iostream>
#include <vector>
#define k 3
#define n 14   // choice "k" object in vector(1 ~ n-1) 

using namespace std;
vector<int> b ;

void print(vector<int> b) {
	for(int i : b) cout << i << " ";
	cout << "\n";
}

void combi(int start, vector<int> b) {
	if (b.size() == k) {
		print(b);
		return;
	}

	for (int i = start + 1; i < n; i++) 
	{
		b.push_back(i);
		combi(i, b);
		b.pop_back();
	}
	return; 
}

int main(void){
	combi(-1,b);
	return 0;
}

