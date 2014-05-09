/*
 * Simplex.cpp
 *
 *  Created on: May 7, 2014
 *      Author: john
 */

#include "Simplex.h"
using namespace std;

Simplex::Simplex() {
	// TODO Auto-generated constructor stub
//	printf("constructor called\n");
	init();
	eval();
}

Simplex::Simplex(double (*func)(Vector3d&)) {
	this->func = func;
	init();
	eval();
}

Simplex::~Simplex() {
	// TODO Auto-generated destructor stub
}
Vector3d Simplex::findOptimal(int max) {
	Vector3d x1, x2, center;

	double y1, a, b, c;
	a = .25;
	b = 1.15;
	c = .35;
	for (int i = 0; i < max && !isConvergent(); ++i) {
//		cout << i << endl;
		order();
		center = getCenter(X[asc[0]], X[asc[1]], X[asc[2]]);
		x1 = center * (1 + a) - X[asc[3]] * a;
		y1 = evalVector(x1);
		if (y1 < Y[asc[0]]) {
			x2 = x1 * (1 + b) - center * b;
			if (evalVector(x2) < Y[asc[0]])
				X[asc[3]] = x2;
			else
				X[asc[3]] = x1;
		} else if (y1 > Y[asc[1]]) {
			if (y1 <= Y[asc[3]])
				X[asc[3]] = x1;
			x2 = X[asc[3]] * c + center * (1 - c);
			if (evalVector(x2) > Y[asc[3]])
				contraction();
			else
				X[asc[3]] = x2;
		} else
			X[asc[3]] = x1;
	}

	return X[asc[0]];
}

void Simplex::init() {
	X[0] = Vector3d::Zero();
	for (int i = 1; i < 4; ++i) {
		X[i] = Vector3d::Zero();
		X[i][(i - 1) % 3] = 1.0;
	}
}

void Simplex::order() {
	asc.clear();
	eval();
	for (int i = 0; i < 4; ++i) {
		if (asc.size() == 0)
			asc.push_back(i);
		else {
			vector<int>::iterator best = asc.begin();
			for (vector<int>::iterator it = asc.begin(); it != asc.end();
					++it) {
				if (Y[i] < Y[*it])
					best = it;
				else if (Y[i] > Y[*it])
					++best;
			}
			asc.insert(best, i);
		}
	}
//	for (vector<int>::iterator it = asc.begin(); it != asc.end(); ++it) {
//		cout << "X_" << *it << " => " << Y[*it] << endl;
//	}
//	cout << endl;
}

void Simplex::transform() {
}

bool Simplex::isConvergent() {
// calculate all lines length
	Vector3d line[6];
	line[0] = X[0] - X[1];
	line[1] = X[0] - X[2];
	line[2] = X[0] - X[3];
	line[3] = X[2] - X[1];
	line[4] = X[3] - X[2];
	line[5] = X[1] - X[3];

	return (line[0].norm() < delta) && (line[1].norm() < delta)
			&& (line[2].norm() < delta) && (line[3].norm() < delta)
			&& (line[4].norm() < delta) && (line[5].norm() < delta);
}

void Simplex::eval() {
	if (func != NULL)
		for (int i = 0; i < 4; ++i) {
			Y[i] = (*func)(X[i]);
		}
	else {
		printf("You haven't pass any heuristic function yet\n");
		exit(1);
	}
}

void Simplex::showDetail() {

	for (int i = 0; i < 4; ++i) {
		cout << "X :" << X[i] << "\nY :" << Y[i] << "\n\n";
	}
	if (isConvergent())
		cout << "convergent\n";
	else
		cout << "not yet convergent\n";
	cout << endl;
}

Vector3d Simplex::reflection() {
	double a = .25;
	Vector3d center = getCenter(X[asc[0]], X[asc[1]], X[asc[2]]);

	cout << X[asc[3]] << endl;

	center = center * (1 + a) - X[asc[3]] * a;

	cout << center << endl;
	return center;
//	int updatedID = asc[3];
//	order();
//	if (asc[0] == updatedID)
//		X[asc[0]] = moveToward(X[updatedID], X[asc[0]]) * 1.5;
//	else if (asc[3] == updatedID)
//		X[asc[3]] = getMid(X[asc[3]], center);
//
//	for (int i = 0; i < 4; ++i) {
//		cout << X[i] << "\n\n";
//	}
}

void Simplex::contraction() {
	order();
	for (int i = 1; i < 4; ++i) {
		X[asc[i]] = getMid(X[asc[0]], X[asc[i]]);
	}

}

void Simplex::showConvergence() {
	Vector3d line[6];
	line[0] = X[0] - X[1];
	line[1] = X[0] - X[2];
	line[2] = X[0] - X[3];
	line[3] = X[2] - X[1];
	line[4] = X[3] - X[2];
	line[5] = X[1] - X[3];
	for (int i = 0; i < 6; ++i) {
		printf("Line %d: %f, ", i, line[i].norm());
	}
	cout << "\n\n";
}

Vector3d Simplex::expansion(Vector3d & x) {
	double y = 1.15;
	Vector3d center = getCenter(X[asc[0]], X[asc[1]], X[asc[2]]);
	center = x * (1 + y) + center * y;
	return center;
}

Vector3d Simplex::getMid(Vector3d & x, Vector3d & y) {
	Vector3d retVal = (x + y) / 2;
	return retVal;
}

Vector3d Simplex::getCenter(Vector3d & x, Vector3d & y, Vector3d & z) {
	Vector3d retVal = (x + y + z) / 3;
//	cout << retVal << endl;
	return retVal;

}

Vector3d Simplex::moveToward(Vector3d & x, Vector3d & y) {
	Vector3d retVal = y - x;
//	cout << retVal << endl;
	return retVal;
}

double Simplex::evalVector(Vector3d& x) {
	return (*func)(x);
}
