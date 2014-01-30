#include "Spheres_electric_monopole.h"
//
//
//
Solver::Spheres_electric_monopole::Spheres_electric_monopole(): 
  Expression(),
  I_( 0. ), r0_values_norm_(0.)
{
  //
  //
  r0_values_.resize(3);
  //
  r0_values_[0] = 0.;
  r0_values_[1] = 0.;
  r0_values_[2] = 0.;
  //
  r_sphere_[0] = 92.; // scalp (medium 1)
  r_sphere_[1] = 86.; // skull (medium 2)
  r_sphere_[2] = 80.; // CSF   (medium 3)
  r_sphere_[3] = 78.; // brain (medium 4)
  //
  sigma_[0][0] = 0.33;   // conductivity longitudinal medium 1
  sigma_[0][1] = 0.33;   // conductivity longitudinal medium 1
  sigma_[1][0] = 0.0042; // conductivity longitudinal medium 2
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
      // WARNING M_[n][0] = M_{j = 1, j-1 = 0} does not exist 
      M_[n][0].Zero();
    
      //
      //
      for (int j = 0 ; j < NUM_SPHERES ; j++ )
	{
	  // exposant
	  nu_[n][j]  = sqrt(1 + 4 * n * (n+1) * sigma_[j][0] / sigma_[j][1]) - 1;
	  nu_[n][j] /= 2.; 

	  // Transfere matrix M_{j,j-1}
	  if ( j > 0)
	    {
	      //
	      // Coefficients for matrix elements
	      M_coeff  = 1.;
	      M_coeff /= (2 * nu_[n][j] + 1) * sigma_[j][0];
	      //
	      lambda   = sigma_[j-1][0] / sigma_[j][0];

	      //
	      // Transfer matrix coefficients
	      M00  = M_coeff * pow( r_sphere_[j], nu_[n][j-1] );
	      M00 /= pow( r_sphere_[j], nu_[n][j] );
	      M00 *= (nu_[n][j] + 1) + lambda * nu_[n][j-1];
	      //
	      M01  = - M_coeff / pow( r_sphere_[j], nu_[n][j] + nu_[n][j-1] + 1 );
	      M01 *= (nu_[n][j-1] + 1) * lambda - (nu_[n][j] + 1);
	      //
	      M10  = - M_coeff * pow( r_sphere_[j], nu_[n][j] + nu_[n][j-1] + 1 );
	      M10 *= nu_[n][j-1] * lambda - nu_[n][j];
	      //
	      M11  = M_coeff * pow( r_sphere_[j], nu_[n][j] );
	      M11 /= pow( r_sphere_[j], nu_[n][j-1] );
	      M11 *= (nu_[n][j-1] + 1) * lambda + nu_[n][j];

	      //
	      // Matrix transfere
	      M_[n][j] <<
		M00, M01,
		M10, M11;
	    }
	}

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
	      A_B_[n][0][2] <<
		(nu_[n][j] + 1) / nu_[n][j] / pow( r_sphere_[0], 2 * nu_[n][j] + 1 ),
		1.;
	    }
	  else
	    {
	      //
	      // A_{j}^{(1,2)}  B_{j}^{(1,2)}
	      A_B_[n][NUM_SPHERES - 1 - j][1] = M_[n][NUM_SPHERES - j].inverse() * A_B_[n][NUM_SPHERES - j][1] ;
	      //
	      A_B_[n][j][2] = M_[n][j] * A_B_[n][j-1][2];
	    }
	}

      //
      // Radial function coefficient
      R_coeff_[n]  = - 1. / (2*n + 1) / sigma_[NUM_SPHERES - 1][0];
      R_coeff_[n] /= A_B_[n][0][1][1];
    }
}
//
//
//
Solver::Spheres_electric_monopole::Spheres_electric_monopole( double X, double Y, double Z ): 
  Expression(),
  I_( 0. )
{
  //
  //
  r0_values_.resize(3);
  //
  r0_values_[0] = X;
  r0_values_[1] = Y;
  r0_values_[2] = Z;
  //
  r0_values_norm_ = sqrt(X*X + Y*Y + Z*Z);
}
//
//
//
Solver::Spheres_electric_monopole::Spheres_electric_monopole(const Spheres_electric_monopole& that): 
  Expression()
{
  I_              = that.I_;
  r0_values_norm_ = that.r0_values_norm_;
  //
  //
  r0_values_.resize(3);
  //
  r0_values_[0] = that.r0_values_[0];
  r0_values_[1] = that.r0_values_[1];
  r0_values_[2] = that.r0_values_[2];
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
  double 
    tempo_value  = 0.,
    return_value = 1.e6;
  int n = 0;

  //
  //
  while( abs( return_value - tempo_value ) > 1.e-2 )
    {
      // reinitialization
      // and n does not start at zero
      return_value = tempo_value;
      n++;

      //
      tempo_value += (2*n + 1) * R(n,x) * P(n,x);
    } 

  //
  //
  values[0] = return_value;
}
//
//
//
double 
Solver::Spheres_electric_monopole::R(const int n,  const Array<double>& x ) const
{
  //
  //
  double r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

  //
  // r_{j} >= r >= r_{j+1}
  int J = 1.e6;
  for( int j = 0 ; j < NUM_SPHERES ; j++)
    if( r >= r_sphere_[j + 1] && r <= r_sphere_[j])
      J = j;
  // r > r_{0}: beyond the electrodes
  if ( J == 1.e6 )
    J = 0;

  //
  // The injection is done on the scalp
  return   R_coeff_[n] * R(n, J, 1, r_sphere_[0]) * R(n, J, 0, r);
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
Solver::Spheres_electric_monopole::P(const int n, const Array<double>& r)const
{
  //
  // \cos \theta = \frac{\vec{r} \cdot \vec{r}_{0}}{|r| \times |r_{0}|}
  //
  double norm_r = sqrt( r[0]*r[0] +r[1]*r[1] + r[2]*r[2] );
  // theta represents the angle between \vec{r} and \vec{r}_{0}
  double 
    cos_theta = r[0]*r0_values_[0] +r[1]*r0_values_[1] + r[2]*r0_values_[2];
  cos_theta /= norm_r * r0_values_norm_;
  
  //
  //
  return boost::math::legendre_p(n,cos_theta);
}
//
//
//
Solver::Spheres_electric_monopole&
Solver::Spheres_electric_monopole::operator =( const Spheres_electric_monopole& that )
{
  I_              = that.I_;
  r0_values_norm_ = that.r0_values_norm_;
  //
  //
  r0_values_.resize(3);
  //
  r0_values_[0] = that.r0_values_[0];
  r0_values_[1] = that.r0_values_[1];
  r0_values_[2] = that.r0_values_[2];
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
