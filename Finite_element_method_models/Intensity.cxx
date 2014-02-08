#include "Intensity.h"
//
//
//
Solver::Intensity::Intensity():
  electric_variable_(""), index_( 0 ), I_( 0. ), surface_(0.), radius_(0.),
  label_("no_name")
{
  //
  //
  r0_values_ = Point();
  e_values_  = Point();
  //
  impedance_ = std::complex<double>(0.,0.);

  not_yet_ = true;
}
//
//
//
Solver::Intensity::Intensity(const Intensity& that): 
  electric_variable_(that.electric_variable_), index_(that.index_), 
  I_(that.I_), label_(that.label_),
  r0_values_(that.r0_values_), e_values_(that.e_values_),
  impedance_(that.impedance_), surface_(that.surface_), radius_(that.radius_)
{

  not_yet_ = true;
}
//
//
//
Solver::Intensity::Intensity( std::string Electric_variable, int Index, 
					      std::string Label, double Intensity,
					      Point X, Point V,double Re_z_l, double Im_z_l,
					      double Surface, double Radius): 
  electric_variable_(Electric_variable), index_( Index ), label_( Label ), I_( Intensity ), 
  r0_values_(X), e_values_(V), impedance_( (Re_z_l,Im_z_l) ), surface_(Surface), radius_(Radius) 
{
}
//
//
//
double 
Solver::Intensity::eval( const Array<double>& x, const ufc::cell& cell) const
{
  Point mid_point(x[0], x[1], x[2]);

  if( I_ != 0 )
    {
      if ( r0_values_.squared_distance(mid_point) < 5 * 5  /*&& not_yet_*/ )
	{
//	  std::cout << "##############################" << std::endl;
//	  std::cout << mid_point << " electric_var: " << electric_variable_ << " label: " << label_ << std::endl;
	  //	  not_yet_ = false;
	  return I_;
	}
      else
	return 0.;
    }
  else
    return 0.;
}
//
//
//
Solver::Intensity&
Solver::Intensity::operator =( const Intensity& that )
{
  electric_variable_ = that.electric_variable_;
  index_  = that.index_;
  I_      = that.I_;
  label_  = that.label_;
  //
  //
  r0_values_ = that.get_r0_values_();
  e_values_  = that.get_e_values_();
  //
  impedance_ = that.get_impedance_();
  //
  surface_   = that.get_surface_();
  radius_    = that.get_radius_();
  
  //
  //
  return *this;
}
//
//
//
std::ostream& 
Solver::operator << ( std::ostream& stream, 
		      const Solver::Intensity& that)
{
//  //
//  //
//  stream 
//    << "Dipole source -- index: " << that.get_index_() << " -- " 
//    << "index cell: " << that.get_index_cell_() << "\n"
//    << "Position:"
//    << " (" << that.get_X_() << ", " << that.get_Y_() << ", " << that.get_Z_()  << ") " 
//    << " -- direction: " 
//    << " (" << that.get_VX_()  << ", " << that.get_VY_() << ", " << that.get_VZ_()  << ") \n"
//    << "Intensity: " << that.get_Q_() << std::endl;
//  
//  //
//  //
//  return stream;
};
