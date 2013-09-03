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
      Cte = 1. / (4 * DOLFIN_PI * a0);
    Vector
      e(3),
      r0(3), r(3);

    //
    // Dipole direction
    std::vector<double> e_values;  
    e_values.push_back( 1. ); 
    e_values.push_back( 0. ); 
    e_values.push_back( 0. );
    e.set_local( e_values );

    //
    // Dipole position in [mm]
    std::vector<double> r0_values;  
    r0_values.push_back( 0.0 + 10. ); 
    r0_values.push_back( 0.0 + 20. );
    r0_values.push_back( 0.0 + 60. );
    r0.set_local( r0_values );

    //
    // Mesure position in [mm]
    std::vector<double> r_values;  
    r_values.push_back( x[0] ); 
    r_values.push_back( x[1] );
    r_values.push_back( x[2] );
    r.set_local( r_values );
    // distance in [mm]
    Vector dist(r);
    dist -= r0;
    double norm_dist = dist.norm("l2");

    // 
    // 10^-3 / 10^-9 = 10^6 : [mm] -> [m]
    values[0] = ( norm_dist < DOLFIN_EPS ? 0 : 
		  Cte * Q * e.inner( dist ) / (norm_dist * norm_dist * norm_dist) ) * 10^6;
    //    values[0] = Cte * Q * e.inner( dist ) / (norm_dist * norm_dist * norm_dist);
  }
};
#endif
