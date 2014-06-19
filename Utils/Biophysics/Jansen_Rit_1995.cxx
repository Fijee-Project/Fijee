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
// Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
// it should be a class member!!
// 
static thread_local int local_electrode_;
// 
// 
// 
extern "C" int ode_system(double t, const double y[], double dydt[], void *params)
{
  // 
  // 
  Utils::Biophysics::Jansen_Rit_1995 *alpha;
  alpha = static_cast< Utils::Biophysics::Jansen_Rit_1995 *>(params);

  // 
  // 
  return alpha->ordinary_differential_equations(t, y, dydt);
}
// 
// 
// 
Utils::Biophysics::Jansen_Rit_1995::Jansen_Rit_1995():
  duration_(20000.), impulse_(120.),
  e0_( 2.5 /*s^{-1}*/), r_( 0.56 /*(mV)^{-1}*/), v0_( 6. /*(mV)*/),
  C_( 135. ),
  a_( 100. /*s^{-1}*/), A_( 3.25 /*(mV)*/), b_( 50. /*s^{-1}*/), B_( 22. /*(mV)*/),
  electrode_(0)
{
  // 
  //  Normal distribution: mu = 2.4 mV and sigma = 2.0 mV
  distribution_ = std::normal_distribution<double>(2.4, 2.0);
  
  // 
  // 
  C1_ = C_;
  C2_ = 0.8 * C_;
  C3_ = C4_ = 0.25 * C_;
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
  // Reach oscillation rhythm
  int transient_stage = 0;
  int MAX_TRANSIENT   = 5000000;
  while ( transient_stage++ < MAX_TRANSIENT )
    {
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
  std::cout << "electrode: " << local_electrode_ << std::endl;
  for ( int i = 1 ; i < duration_ ; i++ )
    {
      double ti = i * delta_t + MAX_TRANSIENT * delta_t;
      // solve
      int status = gsl_odeiv2_driver_apply (driver, &t, ti, y);
      // 
      if (status != GSL_SUCCESS)
	{
	  printf ("error, return value=%d\n", status);
	  abort();
	}
      // record statistics, after the transient state
      //      time_potential_.push_back( std::make_tuple(ti - MAX_TRANSIENT * delta_t, y[0]) );
      electrode_rhythm_[local_electrode_].push_back(std::make_tuple(ti - MAX_TRANSIENT * delta_t,
								    y[0], 0.));
    }

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
    if ( !drawn_[local_electrode_][(int)pulse] )
      {
	p_[local_electrode_] = distribution_(generator_);
	drawn_[local_electrode_][(int)pulse] = true;
      }

  // 
  // System of ODE
  DyDt[0]  = Y[3];
  DyDt[3]  = A_ * a_ * sigmoid(Y[1] - Y[2]) - 2*a_*Y[3] - a_*a_*Y[0];
  // 
  DyDt[1]  = Y[4];
  DyDt[4]  = A_ * a_ * ( p_[local_electrode_] + C2_*sigmoid(C1_ * Y[0]) );
  DyDt[4] += - 2*a_*Y[4] - a_*a_*Y[1];
  // 
  DyDt[2]  = Y[5];
  DyDt[5]  = B_ * b_ * C4_*sigmoid(C3_ * Y[0]) - 2*b_*Y[5] - b_*b_*Y[2];

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
  // 
  int number_of_electrodes = electrode_mapping_.size();

  //
  // R header: alpha rhythm
  // 
  output_stream_
    << "time ";
  // 
  for (int electrode = 0 ; electrode < number_of_electrodes ; electrode++)
    output_stream_ <<  electrode_mapping_[electrode] << " ";
  // 
  output_stream_ << std::endl;

  // 
  // R values: alpha rhythm
  // 
  std::vector< std::list< std::tuple< double, double, double > >::const_iterator > 
    it(number_of_electrodes);
  // 
  for( int electrode = 0 ; electrode < number_of_electrodes ; electrode++ )
    it[electrode] = electrode_rhythm_[electrode].begin();
  // 
  while( it[0] !=  electrode_rhythm_[0].end())
    {
      for (int electrode = 0 ; electrode < number_of_electrodes ; electrode++)
	{
	  // get the time
	  if ( electrode == 0 )
	    output_stream_ <<  std::get<0>( *(it[electrode]) ) << " ";
	  // 
	  
	    output_stream_ <<  std::get<1>( *(it[electrode]++) ) << " ";
	}
      // 
      output_stream_ << std::endl;
    }

  //
  //
  Make_output_file("alpha_rhythm.frame");


  // 
  // FFT
  // 

  //
  // R header: Power spectral density
  // 
  output_stream_
    << "Hz ";
  // 
  for (int electrode = 0 ; electrode < number_of_electrodes ; electrode++)
    output_stream_ <<  electrode_mapping_[electrode] << " ";
  // 
  output_stream_ << std::endl;

  // 
  // R values: Power spectral density
  // 
  int 
    N = 1024, /* N is a power of 2 */
    n = 0;
  // real and imaginary
  double data[2*1024/*N*/];
  std::vector< double* > data_vector(number_of_electrodes);
  // 
  for ( int electrode = 0 ; electrode < number_of_electrodes ; electrode++ )
    {
      n = 0;
      data_vector[electrode] = new double[2*1024/*N*/];
      for( auto time_potential : electrode_rhythm_[electrode] )
	{
	  if ( n < N )
	    {
	      REAL(data_vector[electrode],n)   = std::get<1>(time_potential);
	      IMAG(data_vector[electrode],n++) = 0.0;
	    }
	  else
	    break;
	}
    }

  // 
  // Forward FFT
  // A stride of 1 accesses the array without any additional spacing between elements. 
  for ( auto electrode : data_vector )
    gsl_fft_complex_radix2_forward (electrode, 1/*stride*/, 1024);

  //
  // 
  for ( int i = 0 ; i < N ; i++ )
    {
      output_stream_ 
	<< i << " ";
      // 
      for ( auto electrode : data_vector )
	output_stream_ 
	  << (REAL(electrode,i)*REAL(electrode,i) + IMAG(electrode,i)*IMAG(electrode,i)) / N
	  << " ";
	  
      // 
      output_stream_ << std::endl;
    }
//  // 
//  // 
//  int 
//    N = 1024, /* N is a power of 2 */
//    n = 0;
//  // real and imaginary
//  double data[2*1024/*N*/];
//  // 
//  for( auto time_potential : time_potential_ )
//    {
//      if ( n < N )
//	{
//	  REAL(data,n)   = std::get<1>(time_potential);
//	  IMAG(data,n++) = 0.0;
//	}
//    }
//
//  // 
//  // Forwar FFT
//  // A stride of 1 accesses the array without any additional spacing between elements. 
//  gsl_fft_complex_radix2_forward (data, 1/*stride*/, 1024);
//
//  // 
//  for ( int i = 0 ; i < N ; i++ )
//    output_stream_ 
//      << i << " " 
//      << (REAL(data,i)*REAL(data,i) + IMAG(data,i)*IMAG(data,i)) / N
//      << std::endl;

  //
  //
  Make_output_file("PSD.frame");
#endif
#endif      
}
//
//
//
void 
Utils::Biophysics::Jansen_Rit_1995::init()
{
  // 
  // Initializations
  drawn_.resize( get_number_of_physical_events() );
  //
  for ( int i = 0 ; i < get_number_of_physical_events() ; i++ )
    drawn_[i] = std::vector<bool>(1000000,false);

  //
  p_.resize( get_number_of_physical_events() );
  // 
  for( int i = 0 ; i < get_number_of_physical_events() ; i++ )
    p_[i] = distribution_(generator_);
}
//
//
//
void 
Utils::Biophysics::Jansen_Rit_1995::operator () ()
{
  //
  // Mutex the electrode poping process
  //
  try 
    {
      // lock the electrode
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      // 
      local_electrode_ = electrode_++;
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }

  // 
  // Generate alpha rhythm for electrode local_electrode_
  modelization();
}
// 
// 
// 
void 
Utils::Biophysics::Jansen_Rit_1995::output_XML()
{
  // 
  // Statistic analysise
  Make_analysis();
}
