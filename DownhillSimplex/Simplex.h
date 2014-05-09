/*
 * Simplex.h
 *
 *  Created on: May 7, 2014
 *      Author: john
 */

#ifndef SIMPLEX_H_
#define SIMPLEX_H_
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

class Simplex {
public:
	Simplex();
	Simplex(double (*func)(Vector3d &));
	Simplex(double &d, double (*func)(Vector3d &));
	virtual ~Simplex();
	Vector3d findOptimal(int max);
	void init();
	void eval();
	void order();
	void transform();
	Vector3d reflection();
	void contraction();
	void showConvergence();
	Vector3d expansion(Vector3d &x);
	Vector3d moveToward(Vector3d &x, Vector3d &y);
	Vector3d getCenter(Vector3d &x, Vector3d &y, Vector3d &z);
	Vector3d getMid(Vector3d &x, Vector3d &y);
	bool isConvergent();
	void showDetail();
	double evalVector(Vector3d &x);

	void setDelta(double d) {
		this->delta = d;
	}

private:
	double delta = 1E-5;
	vector<int> asc;
	Vector3d X[4];
	double Y[4];
	double (*func)(Vector3d &)=NULL;
};

#endif /* SIMPLEX_H_ */
