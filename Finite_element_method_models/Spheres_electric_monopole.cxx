#include "Spheres_electric_monopole.h"
//
//
//
Solver::Spheres_electric_monopole::Spheres_electric_monopole(): 
  Expression(),
  I_( 0. )
{}
//
//
//
Solver::Spheres_electric_monopole::Spheres_electric_monopole( const double I, const Point& Injection ): 
  Expression(),
  I_( I )
{
  //
  //
  //
  r0_values_ = Point( Injection.x()*1.e-3, Injection.y()*1.e-3, Injection.z()*1.e-3 );
  //
  //
  r_sphere_[0] = r0_values_.norm(); // scalp (medium 1) r = 92mm
  r_sphere_[1] = 86. * 1.e-3;               // skull (medium 2)
  r_sphere_[2] = 80. * 1.e-3;               // CSF   (medium 3)
  r_sphere_[3] = 78. * 1.e-3;               // brain (medium 4)
  //
  sigma_[0][0] = 0.33;   // conductivity longitudinal medium 1
  sigma_[0][1] = 0.33;   // conductivity longitudinal medium 1
  //  sigma_[1][0] = 0.0042; // conductivity longitudinal medium 2
  sigma_[1][0] = 0.042; // conductivity longitudinal medium 2
  sigma_[1][1] = 0.042;  // conductivity longitudinal medium 2
  sigma_[2][0] = 1.79;   // conductivity longitudinal medium 3
  sigma_[2][1] = 1.79;   // conductivity longitudinal medium 3
  sigma_[3][0] = 0.33;   // conductivity longitudinal medium 4
  sigma_[3][1] = 0.33;   // conductivity longitudinal medium 4

  //
  // Construction of the n first sum elements
  double 
    M00 = 0., M01 = 0.,
    M10 = 0., M11 = 0.;
  double 
    lambda  = 0.,
    M_coeff = 0.;

  //
  // Initialization of terms 
  for ( int j = 0 ; j < NUM_SPHERES ; j++ )
    {
      //
      nu_[0][j] = 0.;

      //
      //
      M_[0][j] <<
	0., 0.,
	0., 0.;
      //
      A_B_[0][j][0].Zero();
      A_B_[0][j][1].Zero();
    }

  //
  // Creation of terms
  // we don't count the first term n = 0
  for (int n = 1 ; n < NUM_ITERATIONS ; n++)
    {
      // WARNING M_[n][0] = M_{j = 0, j-1 = 0} does not exist 
      M_[n][0].Zero();
    
      //
      //
      for (int j = 0 ; j < NUM_SPHERES ; j++ )
	{
	  // exposant
	  nu_[n][j]  = sqrt(1 + 4 * n * (n+1) * sigma_[j][1] / sigma_[j][0]) - 1;
	  nu_[n][j] /= 2.; 

	  // Transfere matrix M_{j,j-1}
	  if ( j > 0)
	    {
	      //
	      // Coefficients for matrix elements
	      M_coeff  = 1.;
	      M_coeff /= (2 * nu_[n][j] + 1);
	      //
	      lambda   = sigma_[j-1][0] / sigma_[j][0];

	      //
	      // Transfer matrix coefficients
	      M00  = M_coeff * pow( r_sphere_[j], nu_[n][j-1] - nu_[n][j] );
	      M00 *= (nu_[n][j] + 1) + lambda * nu_[n][j-1];
	      //
	      M01  = - M_coeff / pow( r_sphere_[j], nu_[n][j] + nu_[n][j-1] + 1 );
	      M01 *= (nu_[n][j-1] + 1) * lambda - (nu_[n][j] + 1);
	      //
	      M10  = - M_coeff * pow( r_sphere_[j], nu_[n][j] + nu_[n][j-1] + 1 );
	      M10 *= nu_[n][j-1] * lambda - nu_[n][j];
	      //
	      M11  = M_coeff * pow( r_sphere_[j], nu_[n][j] - nu_[n][j-1] );
	      M11 *= (nu_[n][j-1] + 1) * lambda + nu_[n][j];

	      //
	      // Matrix transfere
	      M_[n][j] <<
		M00, M01,
		M10, M11;
	      
//	      std::cout << "Mcoeff = " << M_coeff << " r_sphere_[" << j << "]: " << r_sphere_[j] 
//			<< " lambda: " << lambda << "\n"
//			<< " nu_[" << n << "][" << j-1 << "]: " << nu_[n][j-1] 
//			<< " nu_[" << n << "][" << j << "]: " << nu_[n][j] 
//			<< " pow( r_sphere_[j], nu_[n][j-1] - nu_[n][j] ): " 
//			<< pow( r_sphere_[j], nu_[n][j-1] - nu_[n][j] )
//			<< std::endl;
//	      std::cout << "M00 = " << M00 << std::endl;
//	      std::cout << "M01 = " << M01 << std::endl;
//	      std::cout << "M10 = " << M10 << std::endl;
//	      std::cout << "M11 = " << M11 << std::endl;
	    }
	}


//      //      std::cout << "n = " << n << " nu: " << 
//      for ( int j = 0 ; j < NUM_SPHERES ; j++ )
//	std::cout << "nu: " << nu_[n][j] << " |M| = " << M_[n][j].determinant() << "\nM[" << n << "][" << j << "] = \n" <<  M_[n][j] << std::endl;


      //
      //  Coefficients A_{j}^{(1,2)} and B_{j}^{(1,2)}
      for (int j = 0 ; j < NUM_SPHERES ; j++ )
	{
	  if ( j == 0)
	    {
	      // A_{N}^{(1)}  B_{N}^{(1)}
	      A_B_[n][NUM_SPHERES - 1][0] <<
		1.,
		0.;
	      // A_{1}^{(2)}  B_{1}^{(2)}
	      A_B_[n][0][1] <<
		(nu_[n][0] + 1) / nu_[n][0] / pow( r_sphere_[0], 2 * nu_[n][0] + 1 ),
		1.;
	    }
	  else
	    {
	      //
	      // A_{j}^{(1,2)}  B_{j}^{(1,2)}
	      A_B_[n][NUM_SPHERES - j - 1][0] = M_[n][NUM_SPHERES - j].inverse() * A_B_[n][NUM_SPHERES - j][0] ;
	      //
	      A_B_[n][j][1] = M_[n][j] * A_B_[n][j-1][1];
	    }
	}

      //
      // Radial function coefficient
      R_coeff_[n]  = - 1. / (2*n + 1) / sigma_[NUM_SPHERES - 1][0];
      R_coeff_[n] /= A_B_[n][NUM_SPHERES - 1][1][1];
    }
}
//
//
//
Solver::Spheres_electric_monopole::Spheres_electric_monopole(const Spheres_electric_monopole& that): 
  Expression()
{
  I_ = that.I_;
  r0_values_ = that.r0_values_;
  //
  std::copy(that.r_sphere_, that.r_sphere_ + NUM_SPHERES, r_sphere_);
  std::copy(that.sigma_, that.sigma_ + NUM_SPHERES, sigma_);
  //
  std::copy(that.nu_, that.nu_ + NUM_ITERATIONS, nu_);
  std::copy(that.R_coeff_, that.R_coeff_ + NUM_ITERATIONS, R_coeff_);
  //
  for ( int n = 0 ; n < NUM_ITERATIONS ; n++ )
    for( int j = 0 ; j < NUM_SPHERES ; j ++ )
      {
	A_B_[n][j][0] = that.A_B_[n][j][0];
	A_B_[n][j][1] = that.A_B_[n][j][1];
	//
	M_[n][j] = that.M_[n][j];
      }
}
//
//
//
void 
Solver::Spheres_electric_monopole::eval(Array<double>& values, const Array<double>& x) const
{
  //
  //
  Point evaluation_point(x[0] * 1.e-3, x[1] * 1.e-3, x[2] * 1.e-3);

  //
  //
  double 
    tempo_value  = 1.e+6,
    return_value = 0.;


  //
  //
  //
//  if (++n > NUM_ITERATIONS )
//    {
//      std::cerr << "Not enough iteration asked for the validation process!!" << std::endl;
//      abort();
//    }

     //
  for (int n = 1 ; n < NUM_ITERATIONS ; n++)
    {
    return_value += I_ * (2*n + 1) * R(n,evaluation_point) * P(n,evaluation_point);

//      std::cout << "I_: " << I_
//		<< " n: " << n 
//		<< " R: " << R(n,evaluation_point)
//		<< " P: " << P(n,evaluation_point)
//		<< " return: " << return_value
//		<< std::endl;
    }

  //
  //
  values[0] = return_value;
}
//
//
//
double 
Solver::Spheres_electric_monopole::R(const int n,  const Point& x ) const
{
  //
  //
  double r = x.norm();
  
  //
  // r_{j} >= r >= r_{j+1}
  int j = 0;
  //
  if( r > r_sphere_[1] ) 
    j = 0;
 else if( r <= r_sphere_[1] &&  r > r_sphere_[2] )
    j = 1;
  else if( r <= r_sphere_[2] &&  r > r_sphere_[3] )
    j = 2;
  else
    j = 3;

//  std::cout 
//    << "n: " << n 
//    << " j: " << j
//    << "  r0: " << r0_values_.norm()
//    << "  r: " << r
//    << "\n  R_coeff_[n]: " << R_coeff_[n]
//    << " R(n, J, 1, r0): " << R(n, j, 1, r_sphere_[0])
//    << " R(n, J, 0, r): " << R(n, j, 0, r)
//    << " return: " << R_coeff_[n] * R(n, j, 1, r_sphere_[0]) * R(n, j, 0, r)
//    << std::endl << std::endl;

  //
  // The injection is done on the scalp
  return   R_coeff_[n] * R(n, j, 1, r_sphere_[0]) * R(n, j, 0, r);
}
//
//
//
double 
Solver::Spheres_electric_monopole::R( const int n, const int j,  
				      const int i, const double r ) const
{
  return A_B_[n][j][i][0] * pow(r,nu_[n][j]) + A_B_[n][j][i][1] / pow(r,nu_[n][j] + 1);
}
//
//
//
double 
Solver::Spheres_electric_monopole::P(const int n, const Point& r)const
{
  //
  // \cos \theta = \frac{\vec{r} \cdot \vec{r}_{0}}{|r| \times |r_{0}|}
  //
  double norm_r = r.norm();
  // theta represents the angle between \vec{r} and \vec{r}_{0}
  double 
    cos_theta = r.dot( r0_values_ );
  cos_theta /= r.norm() * r0_values_.norm();
  
  //
  //
  return boost::math::legendre_p(n,cos_theta);
}
//
//
//
double 
Solver::Spheres_electric_monopole::Yn(const int n, const Point& r)const
{
  //
  Point x(1,0,0);
  Point z(0,0,1);
  //
  double norm_r  = r.norm();
  double norm_r0 = r0_values_.norm();
  // cos(theta)
  double 
    cos_theta = r.dot( z );
  cos_theta  /= r.norm();
  //
  double theta = acos(cos_theta);
  //
  double 
    cos_theta0 = r0_values_.dot( z );
  cos_theta0  /= r0_values_.norm();
  //
  double theta0 = acos(cos_theta0);

  //
  // phi
  double phi = r.x() / (r.norm() * sqrt( 1 - cos_theta*cos_theta));
  phi = acos(phi);
  //
  double phi0 = r0_values_.x() / (r0_values_.norm() * sqrt( 1 - cos_theta0*cos_theta0));
  phi0 = acos(phi0);
  //
  std::complex<double> return_value;
  
  //
  //
  for ( int m = 0 ; m <= n ; m++ )
    return_value += boost::math::spherical_harmonic( n,  m, theta, phi)*boost::math::spherical_harmonic( n,  m, theta0, phi0);
  
  //
  //
  return return_value.real();
}
//
//
//
Solver::Spheres_electric_monopole&
Solver::Spheres_electric_monopole::operator =( const Spheres_electric_monopole& that )
{
  I_ = that.I_;
  //
  //
  r0_values_ = that.r0_values_;
  //
  std::copy(that.r_sphere_, that.r_sphere_ + NUM_SPHERES, r_sphere_);
  std::copy(that.sigma_, that.sigma_ + NUM_SPHERES, sigma_);
  //
  std::copy(that.nu_, that.nu_ + NUM_ITERATIONS, nu_);
  std::copy(that.R_coeff_, that.R_coeff_ + NUM_ITERATIONS, R_coeff_);
  //
  for ( int n = 0 ; n < NUM_ITERATIONS ; n++ )
    for( int j = 0 ; j < NUM_SPHERES ; j ++ )
      {
	A_B_[n][j][0] = that.A_B_[n][j][0];
	A_B_[n][j][1] = that.A_B_[n][j][1];
	//
	M_[n][j] = that.M_[n][j];
      }

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Solver::operator << ( std::ostream& stream, 
		      const Solver::Spheres_electric_monopole& that)
{
  //
  //
//  stream 
//    << "Dipole source -- index: " << that.get_index_() << "\n"
//    << "Position:"
//    << " (" << that.get_X_() << ", " << that.get_Y_() << ", " << that.get_Z_()  << ") " 
//    << " -- direction: " 
//    << " (" << that.get_VX_()  << ", " << that.get_VY_() << ", " << that.get_VZ_()  << ") \n"
//    << "Intensity: " << that.get_Q_() << " -- conductivity: " << that.get_a0_() << std::endl;
//  
  //
  //
  return stream;
};
