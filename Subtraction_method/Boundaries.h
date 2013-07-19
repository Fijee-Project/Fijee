#ifndef _BOUNDARY_H
#define _BOUNDARY_H
#include <dolfin.h>

using namespace dolfin;

//
// Sub domain for Dirichlet boundary condition
//
class Periphery : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    //    return ( on_boundary );
    return ( on_boundary  );
  }
};

#endif
