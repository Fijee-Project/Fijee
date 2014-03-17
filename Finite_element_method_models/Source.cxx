#include "Source.h"
//
//
//
Solver::Phi::Phi(): 
  Expression(),
  index_( 0 ), Q_( 0. ), a0_( 0. ), name_("no_name")
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
Solver::Phi::Phi(const Phi& that): 
  Expression()
{
  index_ = that.index_;
  Q_     = that.Q_;
  a0_    = that.a0_;
  name_  = that.name_;
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = that.r0_values_[0];
  r0_values_[1] = that.r0_values_[1];
  r0_values_[2] = that.r0_values_[2];
  //
  e_values_[0] = that.e_values_[0];
  e_values_[1] = that.e_values_[1];
  e_values_[2] = that.e_values_[2];
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
  r0_values_[2] = Z;
  //
  e_values_[0] = VX;
  e_values_[1] = VY;
  e_values_[2] = VZ;
  //
  name_ = "dipole_" + std::to_string(Index);
//    + "_" + std::to_string(Intensity) + "_" 
//    + std::to_string(X)  + "_"  + std::to_string(Y) + "_"  + std::to_string(Z) + "_" 
//    + std::to_string(VX) + "_" + std::to_string(VY) + "_" + std::to_string(VZ) + "_"  
//    + std::to_string(a0_);
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
  //  Vector e(3), r0(3), r(3);
  Eigen::Matrix<double, 3, 1> e, r0, r, dist;
  
  //
  // Dipole direction
  //  e.set_local( e_values_ );
  e << e_values_[0], e_values_[1], e_values_[2];

  //
  // Dipole position in [mm]
  //  r0.set_local( r0_values_ );
  r0 << r0_values_[0], r0_values_[1], r0_values_[2];

  //
  // Mesure position in [mm]
//  std::vector<double> r_values(3);  
//  r_values[0] = x[0]; 
//  r_values[1] = x[1];
//  r_values[2] = x[2];
//  r.set_local( r_values );
  r << x[0], x[1], x[2];
  // distance in [mm]
//  Vector dist(r);
//  dist -= r0;
  dist = r - r0;
  double norm_dist = dist.norm();
  
  // 
  // 10^-3 / 10^-9 = 10^6 : [mm] -> [m]
  values[0] = ( norm_dist < DOLFIN_EPS ? 0 : 
		Cte * Q_ * e.dot( dist ) / (norm_dist * norm_dist * norm_dist) * 1.e+6 );
}
//
//
//
Solver::Phi&
Solver::Phi::operator =( const Phi& that )
{
  index_ = that.index_;
  Q_     = that.Q_;
  a0_    = that.a0_;
  name_  = that.name_;
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = that.get_X_();
  r0_values_[1] = that.get_Y_();
  r0_values_[2] = that.get_Z_();
  //
  e_values_[0] = that.get_VX_();
  e_values_[1] = that.get_VY_();
  e_values_[2] = that.get_VZ_();

  //
  //
  return *this;
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


//
//
//
Solver::Current_density::Current_density(): 
  Expression(3,1),
  index_( 0 ), index_cell_( 0 ), Q_( 0. ), name_("no_name")
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
Solver::Current_density::Current_density(const Current_density& that): 
  Expression(3,1)
{
  index_cell_ = that.index_cell_;
  index_ = that.index_;
  Q_     = that.Q_;
  name_  = that.name_;
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = that.r0_values_[0];
  r0_values_[1] = that.r0_values_[1];
  r0_values_[2] = that.r0_values_[2];
  //
  e_values_[0] = that.e_values_[0];
  e_values_[1] = that.e_values_[1];
  e_values_[2] = that.e_values_[2];
}
//
//
//
Solver::Current_density::Current_density( int Index, int Index_Cell, 
					  double Intensity,
					  double X,  double Y,  double Z, 
					  double VX, double VY, double VZ ): 
  Expression(3,1), index_( Index ), index_cell_( Index_Cell ), Q_( Intensity )
{
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = X;
  r0_values_[1] = Y;
  r0_values_[2] = Z;
  //
  e_values_[0] = VX;
  e_values_[1] = VY;
  e_values_[2] = VZ;
  //
  name_ = "dipole_" + std::to_string(Index);
}
//
//
//
void 
Solver::Current_density::eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
{
  if( index_cell_ == cell.index )
    {
//      Vector e(3)/*, r0(3), r(3)*/;
//      
//      //
//      // Dipole direction
//      e.set_local( e_values_ );
//      
//      //
//      // Dipole position in [mm]
//      r0.set_local( r0_values_ );
//      
//      //
//      // Mesure position in [mm]
//      std::vector<double> r_values(3);  
//      r_values[0] = x[0]; 
//      r_values[1] = x[1];
//      r_values[2] = x[2];
//      r.set_local( r_values );
//      // distance in [mm]
//      Vector dist(r);
//      dist -= r0;
//      double norm_dist = dist.norm("l2");
      
      // 
      // \vec{J}_{source} = \vec{Q} \delta(\vec{dist})
      // When the cell will be selected: the 4 vertices will go through the condition
      values[0] = e_values_[0] * Q_ / 4.;
      values[1] = e_values_[1] * Q_ / 4.;
      values[2] = e_values_[2] * Q_ / 4.;
    }
  else
    {
      values[0] = 0.;
      values[1] = 0.;
      values[2] = 0.;
    }
}
//
//
//
Solver::Current_density&
Solver::Current_density::operator =( const Current_density& that )
{
  index_cell_ = that.index_cell_;
  index_ = that.index_;
  Q_     = that.Q_;
  name_  = that.name_;
  //
  //
  r0_values_.resize(3);
  e_values_.resize(3);
  //
  r0_values_[0] = that.get_X_();
  r0_values_[1] = that.get_Y_();
  r0_values_[2] = that.get_Z_();
  //
  e_values_[0] = that.get_VX_();
  e_values_[1] = that.get_VY_();
  e_values_[2] = that.get_VZ_();

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Solver::operator << ( std::ostream& stream, 
		      const Solver::Current_density& that)
{
  //
  //
  stream 
    << "Dipole source -- index: " << that.get_index_() << " -- " 
    << "index cell: " << that.get_index_cell_() << "\n"
    << "Position:"
    << " (" << that.get_X_() << ", " << that.get_Y_() << ", " << that.get_Z_()  << ") " 
    << " -- direction: " 
    << " (" << that.get_VX_()  << ", " << that.get_VY_() << ", " << that.get_VZ_()  << ") \n"
    << "Intensity: " << that.get_Q_() << std::endl;
  
  //
  //
  return stream;
};
