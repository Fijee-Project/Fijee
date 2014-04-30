#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

int main() {
	Vector3d x[3];
	for (int i = 0; i < 3; ++i) {
		x[i] = Vector3d(i, i + 1, i + 2);
		cout << x[i] << "\n\n";
	}
	vector<Vector3d> my_vec;
	for (int i = 0; i < 20; ++i) {
		my_vec.push_back(Vector3d(1.2*i, 1.5 * i, 2.1*i));
	}
	cout << "my_vec contains:\n";
	for (vector<Vector3d>::iterator it = my_vec.begin(); it != my_vec.end();
			++it)
		cout << *it<<"\n\n";
	cout << '\n';
}
