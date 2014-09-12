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
// WARNING
// Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
// Afterward, it should be a class member!!
// 
static thread_local int local_population_;
static thread_local int local_electrode_;
// 
// 
// 
extern "C" int ode_system_JR(double t, const double y[], double dydt[], void *params)
{
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
  Brain_rhythm( 20000 /*ms*/ ),
  pulse_(90./*pulses per second*/),
  e0_( 2.5 /*s^{-1}*/), r_( 0.56 /*(mV)^{-1}*/), v0_( 6. /*(mV)*/),
  C_( 135. ),
  a_( 100. /*s^{-1}*/), A_( 3.25 /*(mV)*/), b_( 50. /*s^{-1}*/), B_( 22. /*(mV)*/),
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
  gsl_odeiv2_system sys = {ode_system_JR, NULL /*jacobian*/, 6, this};
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
  int MAX_TRANSIENT   = 500000;
  while ( transient_stage++ < MAX_TRANSIENT )
    {
      // every second change the noise influence
      //      if ( transient_stage % 1000 )
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
      if ( i % 1000 )
	p_[local_population_] = pulse_ + distribution_(generator_);
      // start after the transient state
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
      local_population_rhythm[2*(i-1)]   = std::to_string(ti - MAX_TRANSIENT * delta_t);
      local_population_rhythm[2*(i-1)]  += std::string(" ");
      local_population_rhythm[2*(i-1)+1] = std::to_string(y[1] - y[2]) + std::string(" ");
      // 
      char_array_size += local_population_rhythm[2*(i-1)].size();
      char_array_size += local_population_rhythm[2*(i-1)+1].size();
      // shift
      population_V_shift_[local_population_] += y[1] - y[2];
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
  free(array_to_compress);
  array_to_compress = nullptr;
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
  // System of ODE
  DyDt[0]  = Y[3];
  DyDt[3]  = A_*a_*sigmoid(Y[1] - Y[2]) - 2*a_*Y[3] - a_*a_*Y[0];
  // 
  DyDt[1]  = Y[4];
  DyDt[4]  = A_*a_*( p_[local_population_] + C2_*sigmoid(C1_ * Y[0]) );
  DyDt[4] += - 2*a_*Y[4] - a_*a_*Y[1];
  // 
  DyDt[2]  = Y[5];
  DyDt[5]  = B_*b_*C4_*sigmoid(C3_ * Y[0]) - 2*b_*Y[5] - b_*b_*Y[2];

  // 
  // 
  return GSL_SUCCESS;
}
// 
// 
//
void
Utils::Biophysics::Jansen_Rit_1995::modelization_at_electrodes()
{
  try
    {
      // 
      // 
      std::cout << "Electrode: " << local_electrode_ << std::endl;
      
      // 
      // initialization of vector of vector (must be all the same size)
      brain_rhythm_at_electrodes_[local_electrode_] = new double[ 2*(duration_-1) ];
      // 
      bool tCS_activated = !parcellation_.empty();
      std::list< std::tuple< double/*time*/, double/*V*/ > >::const_iterator it_tCS_V_ts;

      // 
      // Inflation of data
      std::vector< Bytef >  ts_word;
      std::vector< std::string > ts_values;
      //
      char*  pch     = nullptr;
      Bytef* ts_data = nullptr;
      // 
      Utils::Zlib::Compression inflate;
      for ( int population = 0 ; population < number_samples_ ; population++)
	{
	  // 
	  // 
	  ts_word.clear();
	  ts_values.clear();

	  // 
	  // Inflation of data
	  inflate.in_memory_decompression( population_rhythm_[population], ts_word );
	  // Copy of data
	  int size_of_word = ts_word.size();
	  ts_data = new Bytef[size_of_word];
	  std::copy ( ts_word.begin(), ts_word.end(), ts_data );
	  // 
	  // We lock because of thread collisions
	  {
	    // lock the population
	    std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	    // Cut of data
	    pch = strtok(reinterpret_cast<char*>(ts_data)," ");
	    //
	    while ( pch != nullptr && size_of_word != 0 )
	      {
		size_of_word -= strlen(pch) + 1;
		ts_values.push_back( std::string(pch) );
		pch = strtok (nullptr, " ");
	      }
	  }
	  // 
	  if( static_cast<int>(ts_values.size()) != 2*(duration_-1) )
	    {
	      std::string message = std::string("The size of the inflated data structure is: ")
		+ std::to_string(ts_values.size()) 
		+ std::string(". It should be: ") + std::to_string(2*(duration_-1))
		+ std::string(".");
	      //
	      throw Utils::Error_handler( message,  __LINE__, __FILE__ );
	    }

	  // 
	  // Which parcel belong the population
	  int parcel = populations_[population].get_index_parcel_();
	  if( tCS_activated ) // index parcel starts at 1
	    it_tCS_V_ts = parcellation_[parcel-1].get_V_time_series_().begin(); 


	  // 
	  // Loop over the time series
	  
	  for( int i = 1 ; i < duration_ ; i++ )
	    {
	      // get the time
	      double time = std::stod( ts_values[2*(i-1)] );
	      brain_rhythm_at_electrodes_[local_electrode_][2*(i-1)] = time;
	      // get the potential
	      double V = std::stod( ts_values[2*(i-1) + 1] ) - population_V_shift_[population];
	      V *= 1.e-03; // mv -> V
	      V *= (leadfield_matrix_[local_electrode_].get_V_dipole_())[population];
	      // tCS
	      if( tCS_activated )
		if( it_tCS_V_ts != parcellation_[parcel-1].get_V_time_series_().end() )
		  if( std::get<0>(*it_tCS_V_ts) == time)
		    {
		      V += std::get<1>(*it_tCS_V_ts);
		      it_tCS_V_ts++;
		    }

	      //
	      // 
	      brain_rhythm_at_electrodes_[local_electrode_][2*(i-1) + 1] += V;
	    }

	  // 
	  //
	  delete[] ts_data;
	  ts_data = nullptr;
	}
    }
  catch( Utils::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
}
//
//
//
void 
Utils::Biophysics::Jansen_Rit_1995::init()
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
Utils::Biophysics::Jansen_Rit_1995::operator () ( const Pop_to_elec_type Threading )
{
  //
  // Mutex the population poping process
  //
  try 
    {
      switch(Threading)
	{
	case POP:
	  {
	    // Block for the lock guard
	    {
	      // lock the population
	      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	      // 
	      local_population_ = population_++;
	    }

	    // 
	    // Generate alpha rhythm for population local_population_
	    modelization();

	    // 
	    break;
	  };
	case ELEC:
	  {
	    // Block for the lock guard
	    {
	      // lock the population
	      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	      // 
	      local_electrode_ = electrode_++;
	    }

	    // 
	    // Generate alpha rhythm for electrode local_electrode_
	    modelization_at_electrodes();

	    // 
	    break;
	  };
	default:
	    {
	      std::string message = std::string("You are asking for the wrong Threading type: ");
	      throw Utils::Exit_handler( message, 
					 __LINE__, __FILE__ );
	    };
	}
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }
  catch( Utils::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
}
