#ifndef _SOURCES_H
#define _SOURCES_H
#include <dolfin.h>
#include <vector>

using namespace dolfin;

//
// Potential Phi0
//
class Phi : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double 
      Q  = 0.000000001 ,
      a0 = 0.33,
      Cte = - 1. / (4 * DOLFIN_PI * a0);
    Vector
      e(2),
      r0(2), r(2);
    //
    std::vector<double> e_values;  
    e_values.push_back( 1. ); e_values.push_back( 0. );
    e.set_local( e_values );

    std::vector<double> r0_values;  
    r0_values.push_back( 0.075 ); r0_values.push_back( 0. );
    r0.set_local( r0_values );

    std::vector<double> r_values;  
    r_values.push_back( x[0] ); r_values.push_back( x[1] );
    r.set_local( r_values );
    //
    Vector dist(r);
    dist -= r0;
    double norm_dist = dist.norm("l2");
    //
    values[0] = ( norm_dist < DOLFIN_EPS ? 0 : 
		  Cte * Q * e.inner( dist ) / (norm_dist * norm_dist * norm_dist) );
    //    values[0] = Cte * Q * e.inner( dist ) / (norm_dist * norm_dist * norm_dist);
  }
};
#endif
