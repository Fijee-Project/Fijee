//============================================================================
// Name        : DHSimplex.cpp
// Author      : JOhn He
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "Simplex.h"

using namespace std;
inline double banana_helper(const double& x, const double& y) {
	return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}
double banana(Vector3d& point) {
	int x = point[0];
	int y = point[1];
	int z = point[2];
	return banana_helper(x, y) + banana_helper(y, z);
}

int main() {
//	Vector3d v3(1, 0, 0);
//	Vector3d v1(0, 0, 0);
//	Vector3d vv = v1 - v3;
//
//	vv.norm();
//	cout<<vv.norm()<<endl;
//	cout << banana(v3) << endl;

//	Simplex n;
//	n.showDetail();

	Simplex *sx;
	sx = new Simplex(banana);
//	sx->showDetail();
	Vector3d vc = sx->findOptimal(200);
	cout << vc << endl;
//	for (int i = 0; i < 10; ++i) {
//		sx->reflection();
//	}
//
//	sx->showConvergence();
//	sx->showDetail();

//	sx->contraction();
//	sx->showConvergence();
//	sx->showDetail();
//	sx->contraction();
//	sx->showConvergence();
//	sx->showDetail();
//	sx->order();

//	v3 = sx.findOptimal();
//	cout<<v3<<endl;
	return 0;
}
