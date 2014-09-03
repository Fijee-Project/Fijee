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
#include "Wendling_2002.h"
// 
// WARNING
// Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
// Afterward, it should be a class member!!
// 
static thread_local int local_population_;
// 
// 
// 
extern "C" int ode_system_W02(double t, const double y[], double dydt[], void *params)
{
  // 
  Utils::Biophysics::Wendling_2002 *alpha;
  alpha = static_cast< Utils::Biophysics::Wendling_2002 *>(params);

  // 
  // 
  return alpha->ordinary_differential_equations(t, y, dydt);
}
// 
// 
// 
Utils::Biophysics::Wendling_2002::Wendling_2002():
  Brain_rhythm( 20000 /*ms*/ ),
  pulse_(90./*pulses per second*/),
  e0_( 2.5 /*s^{-1}*/), r_( 0.56 /*(mV)^{-1}*/), v0_( 6. /*(mV)*/),
  C_( 135. ),
  a_( 100. /*s^{-1}*/), A_( 3.25 /*(mV)*/), b_( 50. /*s^{-1}*/), B_( 22. /*(mV)*/),
  g_(500. /*s^{-1}*/), G_( 10. /*(mV)*/),
  population_(0)
{
  // 
  // p(t) = <p> + \varepsilon; 
  // <p> = pulse_ and \varepsilon \sim \mathcal{N}(0., 30.)
  distribution_ = std::normal_distribution<double>(0., 30.);
  // 
  C1_ = C_;
  C2_ = 0.8 * C_;
  C3_ = C4_ = 0.25 * C_;
  C5_ = 0.3 * C_;
  C6_ = 0.1 * C_;
  C7_ = 0.8 * C_;
}
// 
// 
//
void
Utils::Biophysics::Wendling_2002::modelization()
{
  // 
  // Runge-Kutta
  // 
  // Create the system of ode
  gsl_odeiv2_system sys = {ode_system_W02, NULL /*jacobian*/, 10, this};
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
  double y[10] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // 
  // Reach oscillation rhythm
  int transient_stage = 0;
  int MAX_TRANSIENT   = 500000;
  while ( transient_stage++ < MAX_TRANSIENT )
    {
      // every second change the noise influence
      if ( transient_stage % 1000 )
	p_[local_population_] = pulse_ + distribution_(generator_);
      //
      double ti = transient_stage * delta_t;
      // solve
      int status = gsl_odeiv2_driver_apply (driver, &t, ti, y);
     // 
      if (status != GSL_SUCCESS)
	{
	  printf ("error, return value=%d\n", status);
	  abort();
	}
    }
  
  //
  // 
  std::cout << "population: " << local_population_ << std::endl;
  std::string* local_population_rhythm = new std::string[2*(duration_-1)];
  int          char_array_size = 0;
  //
  for ( int i = 1 ; i < duration_ ; i++ )
    {
      // every second change the noise influence
      //      if ( i % 1000 )
      p_[local_population_] = pulse_ + distribution_(generator_);
      // 
      double ti = i * delta_t + MAX_TRANSIENT * delta_t;
      // solve
      int status = gsl_odeiv2_driver_apply (driver, &t, ti, y);
      // 
      if (status != GSL_SUCCESS)
	{
	  std::cerr << "error, return value = " << status << std::endl;
	  abort();
	}
      // record statistics, after the transient state
      local_population_rhythm[2*(i-1)]   = std::to_string(ti - MAX_TRANSIENT * delta_t);
      local_population_rhythm[2*(i-1)]  += std::string(" ");
      local_population_rhythm[2*(i-1)+1] = std::to_string(y[1] - y[2] - y[3]) + std::string(" ");
      // 
      char_array_size += local_population_rhythm[2*(i-1)].size();
      char_array_size += local_population_rhythm[2*(i-1)+1].size();
      // shift
      population_V_shift_[local_population_] += y[1] - y[2] - y[3];
    }
  // average the shift
  population_V_shift_[local_population_] /= static_cast<double>(duration_);

  // 
  // Convert strings into an array of char
  char* array_to_compress = (char*)malloc( char_array_size*sizeof(char) );
  // first occurence
  strcpy( array_to_compress, local_population_rhythm[0].c_str() );
  //
  for( int str_idx = 1 ; str_idx < 2*(duration_ - 1) ; str_idx++ )
    strcat(array_to_compress, local_population_rhythm[str_idx].c_str());
  //
  Utils::Zlib::Compression deflate;
  deflate.in_memory_compression( array_to_compress, char_array_size, 
				 population_rhythm_[local_population_] );

  // 
  // Clean area
  delete[] local_population_rhythm;
  local_population_rhythm = nullptr;
  delete[] array_to_compress;
  array_to_compress = nullptr;
  // 
  gsl_odeiv2_driver_free (driver);
}
// 
// 
//
int
Utils::Biophysics::Wendling_2002::ordinary_differential_equations( double T, const double Y[], double DyDt[] )
{
  // 
  // System of ODE
  DyDt[0]  = Y[5];
  DyDt[5]  = A_*a_*sigmoid(Y[1] - Y[2] - Y[3]) - 2*a_*Y[5] - a_*a_*Y[0];
  // 
  DyDt[1]  = Y[6];
  DyDt[6]  = A_*a_*( p_[local_population_] + C2_*sigmoid(C1_*Y[0]) );
  DyDt[6] += - 2*a_*Y[6] - a_*a_*Y[1];
  // 
  DyDt[2]  = Y[7];
  DyDt[7]  = B_*b_*C4_*sigmoid(C3_*Y[0]) - 2*b_*Y[7] - b_*b_*Y[2];
  // 
  DyDt[3]  = Y[8];
  DyDt[8]  = G_*g_*C7_*sigmoid(C5_*Y[0] - C6_*Y[4]) - 2*g_*Y[8] - g_*g_*Y[3];
  // 
  DyDt[4]  = Y[9];
  DyDt[9]  = B_*b_*C6_*sigmoid(C3_*Y[0]) - 2*b_*Y[9] - b_*b_*Y[4];

  // 
  // 
  return GSL_SUCCESS;
}
//
//
//
void 
Utils::Biophysics::Wendling_2002::init()
{
  // 
  // Initializations
  // 

  // 
  //
  p_.resize( get_number_of_physical_events() );
  // 
  for( int i = 0 ; i < get_number_of_physical_events() ; i++ )
    p_[i] = pulse_ + distribution_(generator_);
}
//
//
//
void 
Utils::Biophysics::Wendling_2002::operator () ()
{
  //
  // Mutex the population poping process
  //
  try 
    {
      // lock the population
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      // 
      local_population_ = population_++;
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }

  // 
  // Generate alpha rhythm for population local_population_
  modelization();
}
