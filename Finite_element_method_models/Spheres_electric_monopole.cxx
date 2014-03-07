#include "Spheres_electric_monopole.h"
//
//
double Pr( double r, double nu )
{
  return pow( r, nu );
}
double PPr( double r, double nu )
{
  return nu * pow( r, nu - 1 );
}
double Qr( double r, double nu )
{
  return pow( r, -(nu+1) );
}
double QPr( double r, double nu )
{
  return -(nu+1) * pow( r, -(nu + 2) );
}


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
  r0_values_ = Point( Injection.x() * 1.e-3, Injection.y() * 1.e-3, Injection.z() * 1.e-3 );
  //
  //
  std::cout << "LALALA\n" << r0_values_ << std::endl;
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
      M_det[0][j] = 0.;
      //
      M_Inv[0][j] <<
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
	      // Determinant
	      double delta;
	      //
	      delta  = ((nu_[n][j]+1) + nu_[n][j-1]*lambda) * ((nu_[n][j-1]+1)*lambda + nu_[n][j]);
	      delta -= (nu_[n][j-1]*lambda - nu_[n][j]) * ((nu_[n][j-1]+1)*lambda - (nu_[n][j]+1));
	      delta /= (2*nu_[n][j] + 1)*(2*nu_[n][j] + 1);
	      //
	      M_det[n][j] = delta;
	      // Inverse
	      M_Inv[n][j] <<
		M11, -M01,
		-M10, M00;
	      //
	      M_Inv[n][j] /= delta;
	      
	      
	      if ( false )
		{
		  Eigen::Matrix<double, 2,2> alpha, beta, M_test;
		  //
		  alpha <<
		    Pr( r_sphere_[j], nu_[n][j] ),               Qr( r_sphere_[j], nu_[n][j] ),
		    sigma_[j][0]*PPr( r_sphere_[j], nu_[n][j] ), sigma_[j][0]*QPr( r_sphere_[j], nu_[n][j] );
		  //
		  beta << 
		    Pr( r_sphere_[j], nu_[n][j-1] ),                 Qr( r_sphere_[j], nu_[n][j-1] ),
		    sigma_[j-1][0]*PPr( r_sphere_[j], nu_[n][j-1] ), sigma_[j-1][0]*QPr( r_sphere_[j], nu_[n][j-1] );
		  //
		  M_test = alpha.inverse() * beta;
		  //
		  std::cout << "transfer matrix m: \n" << M_[n][j] << std::endl;
		  std::cout << "transfer test   m: \n" << M_test << std::endl;
		  
		  //
		  // Determinant
		  double delta;
		  //
		  delta  = ((nu_[n][j]+1) + nu_[n][j-1]*lambda) * ((nu_[n][j-1]+1)*lambda + nu_[n][j]);
		  delta -= (nu_[n][j-1]*lambda - nu_[n][j]) * ((nu_[n][j-1]+1)*lambda - (nu_[n][j]+1));
		  delta /= (2*nu_[n][j] + 1)*(2*nu_[n][j] + 1);
		  //
		  std::cout << "|m|: " << M_[n][j].determinant() << std::endl;
		  std::cout << "delta: " << delta << std::endl;
		  
		  //
		  // Inverse
		  double 
		    M_Inv00, 
		    M_Inv01, 
		    M_Inv10, 
		    M_Inv11;
		  //
		  M_coeff /= delta;
		  //
		  M_Inv00  = M_coeff * pow( r_sphere_[j], nu_[n][j] - nu_[n][j-1] );
		  M_Inv00 *= (nu_[n][j-1] + 1) * lambda + nu_[n][j];
		  //
		  M_Inv01  = M_coeff / pow( r_sphere_[j], nu_[n][j] + nu_[n][j-1] + 1 );
		  M_Inv01 *= (nu_[n][j-1] + 1) * lambda - (nu_[n][j] + 1);
		  //
		  M_Inv10  = M_coeff * pow( r_sphere_[j], nu_[n][j] + nu_[n][j-1] + 1 );
		  M_Inv10 *= nu_[n][j-1] * lambda - nu_[n][j];
		  //
		  M_Inv11  = M_coeff * pow( r_sphere_[j], nu_[n][j-1] - nu_[n][j] );
		  M_Inv11 *= (nu_[n][j] + 1) + lambda * nu_[n][j-1];
		  //
		  Eigen::Matrix< double, 2, 2 > M_Inv;
		  M_Inv << 
		    M_Inv00, M_Inv01, 
		    M_Inv10, M_Inv11;
		  //
		  //
		  std::cout << "transfer matrix m^-1: \n" << M_[n][j].inverse() << std::endl;
		  std::cout << "transfer test   m^-1: \n" << M_Inv << std::endl;
		}

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
	      A_B_[n][NUM_SPHERES - j - 1][0] = M_Inv[n][NUM_SPHERES - j] * A_B_[n][NUM_SPHERES - j][0] ;
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
Solver::Spheres_electric_monopole::eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
{
  //
  //
  Point evaluation_point(x[0] * 1.e-3, x[1] * 1.e-3, x[2] * 1.e-3);
  double point_norm = evaluation_point.norm();
  //
//  if( fabs( point_norm - r_sphere_[3] ) < 0.001 ) point_norm = r_sphere_[3];
//  else if( fabs( point_norm - r_sphere_[2] ) < 0.001 ) point_norm = r_sphere_[2];
//  else if( fabs( point_norm - r_sphere_[1] ) < 0.001 ) point_norm = r_sphere_[1];
//  else if( fabs( point_norm - r_sphere_[0] ) < 0.001 ) point_norm = r_sphere_[0];

  //
  //
  double 
    tempo_value  = 1.e+6,
    return_value = 0.;
  //
  int n = 0;

  //
  //
  while( std::abs(return_value - tempo_value ) > 1.e-4 )
    {
      //
      //
      tempo_value = return_value;
      //
      if (++n > NUM_ITERATIONS )
	{
	  std::cerr << "Not enough iteration asked for the validation process!!" << std::endl;
	  abort();
	}
  
//      std::cout << cell.index
//		<< " " << n 
//		<< " " << evaluation_point.norm()
//		<< " " << r0_values_.norm()
//		<< " " << evaluation_point.dot(r0_values_) /(evaluation_point.norm() * r0_values_.norm());

      //
      return_value += I_ * (2*n + 1) * R(n,point_norm) * P(n,evaluation_point)  / ( 4 * 3.14159 );
   }

     //
      std::cout << " n: " << n 
		<< " r: " << point_norm
		<< " r0: " << r0_values_.norm()
//		<< " cos: " <<  evaluation_point.dot(r0_values_) /(evaluation_point.norm() * r0_values_.norm())
		<< " R: " << R(n,point_norm)
		<< " P: " << P(n,evaluation_point)
//		<< " return: " << return_value
		<< std::endl;
      //      std::cout << "n: " << n << " " << return_value << " " << tempo_value << std::endl;

  //
  //
      values[0] = ( n > 100 ? 0. : return_value );
}
//
//
//
double 
Solver::Spheres_electric_monopole::R(const int n,  const double r ) const
{
  //
  //
  //  double r = x.norm();
  
  //
  // r_{j} >= r >= r_{j+1}
  int j = 0;
  //
  if( r >= r_sphere_[1] ) 
    j = 0;
 else if( r <= r_sphere_[1] &&  r >= r_sphere_[2] )
   {
     if( fabs( r - r_sphere_[1] ) < 0.001 ) 
       {
	 if( R(n, 0, 0, r) <  R(n, 1, 0, r))
	   j = 0;
	 else
	   j = 1;
       }
     else if ( fabs( r - r_sphere_[2] ) < 0.001 )
       {
	 if( R(n, 1, 0, r) <=  R(n, 2, 0, r))
	   j = 1;
	 else
	   j = 2;
       }
     else
       j = 1;
   }
  else if( r <= r_sphere_[2] &&  r >= r_sphere_[3] )
    {
      if( fabs( r - r_sphere_[2] ) < 0.001 ) 
       {
	 if( R(n, 1, 0, r) <  R(n, 2, 0, r))
	   j = 1;
	 else
	   j = 2;
       }
     else if ( fabs( r - r_sphere_[3] ) < 0.001 )
       {
	 if( R(n, 2, 0, r) <=  R(n, 3, 0, r))
	   j = 2;
	 else
	   j = 3;
       }
     else
       j = 2;
    }
  else
    j = 3;

//  std::cout 
//    << " " << j
//    << " " << R_coeff_[n];

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
//  std::cout 
//    << " " << i
//    << " " << A_B_[n][j][i][0] 
//    << " " << pow(r,nu_[n][j])
//    << " " << A_B_[n][j][i][1]
//    << " " << pow(r,- (nu_[n][j] + 1) );
//  
//  if (i == 0)
//    std::cout << std::endl;

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
