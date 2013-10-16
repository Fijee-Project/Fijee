#include "Source.h"
//
//
//
Solver::Phi::Phi(): 
  Expression(),
  index_( 0 ), Q_( 0. ), a0_( 0. )
{
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = 0.;
  r0_values_[1] = 0.;
  r0_values_[2] = 0.;
  //
  e_values_[0] = 0.;
  e_values_[1] = 0.;
  e_values_[2] = 0.;
}
//
//
//
Solver::Phi::Phi( int Index, double Intensity,
	  double X,  double Y,  double Z, 
	  double VX, double VY, double VZ,
	  double L1, double L2, double L3 ): 
  Expression(),
  index_( Index ), Q_( Intensity ), a0_( (L1+L2+L3) / 3. )
{
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = X;
  r0_values_[1] = Y;
  r0_values_[2] = Y;
  //
  e_values_[0] = VX;
  e_values_[1] = VY;
  e_values_[2] = VY;
}
//
//
//
void 
Solver::Phi::eval(Array<double>& values, const Array<double>& x) const
{
  double 
    //     Q  = 0.000000001,
    //     a0 = 0.33,
    Cte = 1. / (4 * DOLFIN_PI * a0_);
  Vector e(3), r0(3), r(3);
  
  //
  // Dipole direction
  //    std::vector<double> e_values;  
  //    e_values.push_back( 1. ); 
  //    e_values.push_back( 0. ); 
  //    e_values.push_back( 0. );
  e.set_local( e_values_ );
  
  //
  // Dipole position in [mm]
  //    std::vector<double> r0_values;  
  //    r0_values.push_back( 0.0 + 10. ); 
  //    r0_values.push_back( 0.0 + 20. );
  //    r0_values.push_back( 0.0 + 60. );
  r0.set_local( r0_values_ );
  
  //
  // Mesure position in [mm]
  std::vector<double> r_values(3);  
  r_values[0] = x[0]; 
  r_values[1] = x[1];
  r_values[2] = x[2];
  r.set_local( r_values );
  // distance in [mm]
  Vector dist(r);
  dist -= r0;
  double norm_dist = dist.norm("l2");
  
  // 
  // 10^-3 / 10^-9 = 10^6 : [mm] -> [m]
  values[0] = ( norm_dist < DOLFIN_EPS ? 0 : 
		Cte * Q_ * e.inner( dist ) / (norm_dist * norm_dist * norm_dist) ) * 1.e+6;
}
//
//
//
std::ostream& 
Solver::operator << ( std::ostream& stream, 
		      const Solver::Phi& that)
{
  //
  //
  stream 
    << "Dipole source -- index: " << that.get_index_() << "\n"
    << "Position:"
    << " (" << that.get_X_() << ", " << that.get_Y_() << ", " << that.get_Z_()  << ") " 
    << " -- direction: " 
    << " (" << that.get_VX_()  << ", " << that.get_VY_() << ", " << that.get_VZ_()  << ") \n"
    << "Intensity: " << that.get_Q_() << " -- conductivity: " << that.get_a0_() << std::endl;
  
  //
  //
  return stream;
};
