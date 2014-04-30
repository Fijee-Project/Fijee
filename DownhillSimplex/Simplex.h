/*
 * Simplex.h
 *
 *  Created on: Apr 29, 2014
 *      Author: john
 */

#ifndef SIMPLEX_H_
#define SIMPLEX_H_
#include <Eigen/Dense>

namespace std {
using Eigen;
class Simplex {
public:
	Simplex();
	virtual ~Simplex();
private:
	Vector3d x[3];
};

} /* namespace std */

#endif /* SIMPLEX_H_ */
