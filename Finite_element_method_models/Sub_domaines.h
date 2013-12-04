#ifndef _SUB_DOMAINES_H
#define _SUB_DOMAINES_H
#include <dolfin.h>

using namespace dolfin;

//
// Sub domain for Dirichlet boundary condition
// r = 78
class Brain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.077 + 0.001 );
  }
};

// r = 80
class CSF : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) > 0.077  && 
	     sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.080 + 0.002 );
  }
};

// r = 86
class Skull : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1)  ) > 0.080 + DOLFIN_EPS && 
	     sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.086  );
  }
};

// r = 92
class Scalp : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) > 0.086  && 
	     sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.092 + DOLFIN_EPS || on_boundary );
  }
};

#endif
