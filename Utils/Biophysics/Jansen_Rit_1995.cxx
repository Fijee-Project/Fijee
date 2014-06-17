//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#include "Jansen_Rit_1995.h"
// 
// 
// 
extern "C" int ode_system(double t, const double y[], double dydt[], void *params)
{
  // 
  // 
  Utils::Biophysics::Jansen_Rit_1995 *alpha;
  alpha = reinterpret_cast< Utils::Biophysics::Jansen_Rit_1995 *>(params);

  // 
  // 
  return alpha->ordinary_differential_equations(t, y, dydt);
}
// 
// 
// 
Utils::Biophysics::Jansen_Rit_1995::Jansen_Rit_1995():
  duration_(20000.), impulse_(320.),
  e0_( 2.5 /*s^{-1}*/), r_( 0.56 /*(mV)^{-1}*/), v0_( 6. /*(mV)*/),
  C_( 135. ),
  a_( 100. /*s^{-1}*/), A_( 3.25 /*(mV)*/), b_( 50. /*s^{-1}*/), B_( 22. /*(mV)*/)
{
  // 
  //  Normal distribution: mu = 2.4 mV and sigma = 2.0 mV
  distribution_ = std::normal_distribution<double>(2.4, 2.0);
  // 
  drawn_ = std::vector<bool>(1000000,false);
  
  // 
  // 
  C1_ = C_;
  C2_ = 0.8 * C_;
  C3_ = C4_ = 0.25 * C_;

  // 
  // 
  p_.reset( new double (distribution_(generator_)) );
}
// 
// 
//
void
Utils::Biophysics::Jansen_Rit_1995::modelization()
{
  // 
  // Runge-Kutta
  // 
  // Create the system of ode
  gsl_odeiv2_system sys = {ode_system, NULL /*jacobian*/, 6, this};
  // Step types
  // Step Type: gsl_odeiv2_step_rk2   - Explicit embedded Runge-Kutta (2, 3) method. 
  // Step Type: gsl_odeiv2_step_rk4   - Explicit 4th order Runge-Kutta. 
  // Step Type: gsl_odeiv2_step_rkf45 - Explicit Runge-Kutta-Fehlberg (4, 5) method. 
  // ...
  gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new ( &sys, 
							      gsl_odeiv2_step_rkf45, 
							      1e-6, 1e-6, 1.);
  //
  // time and initialization
  double 
    t = 0.0,
    delta_t = 1. / 1000.; /* EEG trigges every 1ms */
  // we have 6 unknowns
  double y[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // 
  // 
  for ( int i = 1 ; i < duration_ + 500; i++ )
    {
      double ti = i * delta_t;
      // solve
      int status = gsl_odeiv2_driver_apply (driver, &t, ti, y);
      // 
      if (status != GSL_SUCCESS)
	{
	  printf ("error, return value=%d\n", status);
	  abort();
	}
      // record statistics, after the transient state
      if( i > 500 )
	time_potential_.push_back( std::make_tuple(t, y[0]) );
    }

  // 
  // Statistic analysise
  Make_analysis();

  // 
  // 
  gsl_odeiv2_driver_free (driver);
}
// 
// 
//
int
Utils::Biophysics::Jansen_Rit_1995::ordinary_differential_equations( double T, const double Y[], double DyDt[] )
{
  // 
  // When puse is a multiple of impulse_ we change the amplitude
  double pulse = T * impulse_;
  double pulse_int;
  //
  double rest = std::modf(pulse, &pulse_int);

  // 
  // 
  if( rest == 0. )
    if ( !drawn_[(int)pulse] )
      {
	*p_ = distribution_(generator_);
	drawn_[(int)pulse] = true;
      }

  // 
  // System of ODE
  DyDt[0] = Y[3];
  DyDt[3] = A_ * a_ * this->sigmoid(Y[1] - Y[2]) - 2*a_*Y[3] - a_*a_*Y[0];
  // 
  DyDt[1] = Y[4];
  DyDt[4] = A_ * a_ * ( *p_ + C2_*sigmoid(C1_ * Y[0]) ) - 2*a_*Y[4] - a_*a_*Y[1];
  // 
  DyDt[2] = Y[5];
  DyDt[5] = B_ * b_ * C4_*sigmoid(C3_ * Y[0]) - 2*b_*Y[5] - b_*b_*Y[2];

  // 
  // 
  return GSL_SUCCESS;
}
//
//
//
void 
Utils::Biophysics::Jansen_Rit_1995::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
  //
  //
  output_stream_
    << "time potential "
    << std::endl;

  // 
  for( auto time_potential : time_potential_ )
    output_stream_
      << std::get<0>(time_potential) << " " 
      << std::get<1>(time_potential) 
      << std::endl;

  //
  //
  Make_output_file("alpha_rhythm.frame");
#endif
#endif      
}
