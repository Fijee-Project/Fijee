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
#include "Molaee_Ardekani_Wendling_2009.h"
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
extern "C" int ode_system_MAW(double t, const double y[], double dydt[], void *params)
{
  // 
  Biophysics::Molaee_Ardekani_Wendling_2009 *alpha;
  alpha = static_cast< Biophysics::Molaee_Ardekani_Wendling_2009 *>(params);

  // 
  // 
  return alpha->ordinary_differential_equations(t, y, dydt);
}
// 
// 
// 
Biophysics::Molaee_Ardekani_Wendling_2009::Molaee_Ardekani_Wendling_2009():
  Brain_rhythm( 20000 /*ms*/ ),
  pulse_(90./*pulses per second*/),
  e0P_( 10. /*s^{-1}*/), e0I1_( 10. /*s^{-1}*/), e0I2_( 10. /*s^{-1}*/), 
  rP_( 0.7 /*(mV)^{-1}*/), rI1_( 0.7 /*(mV)^{-1}*/), rI2_( 0.7 /*(mV)^{-1}*/), 
  v0P_( 1. /*(mV)*/), v0I1_( 4. /*(mV)*/), v0I2_( 4. /*(mV)*/),
  CPP_ ( 55. ),   CPI1_( 80. ),   CPI2_( 90. ),  /* Number of synaptic connections */
  CI1P_( 20. ),  CI1I1_( 15. ),   CI2P_( 25. ),  /* Number of synaptic connections */
  CI2I1_( 20. ),                  CI2I2_( 40. ),  /* Number of synaptic connections */  
  a_(  40. /*s^{-1}*/), A_( 5.5 /*(mV)*/), 
  b_(  20. /*s^{-1}*/), B_( 8.  /*(mV)*/),
  g_( 150. /*s^{-1}*/), G_( 10.  /*(mV)*/),
  population_(0)
{
  // 
  // p(t) = <p> + \varepsilon; 
  // <p> = pulse_ and \varepsilon \sim \mathcal{N}(0., 30.)
  distribution_ = std::normal_distribution<double>(0., 30.);
}
// 
// 
//
void
Biophysics::Molaee_Ardekani_Wendling_2009::modelization()
{
  // 
  // Runge-Kutta
  // 
  // Create the system of ode
  gsl_odeiv2_system sys = {ode_system_MAW, NULL /*jacobian*/, 10, this};
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
  int MAX_TRANSIENT   = 500000;//1000000;
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
      local_population_rhythm[2*(i-1)]    = std::to_string(ti - MAX_TRANSIENT * delta_t);
      local_population_rhythm[2*(i-1)]   += std::string(" ");
      local_population_rhythm[2*(i-1)+1]  = std::to_string(y[0] + y[4] - y[1] - y[2]); 
      local_population_rhythm[2*(i-1)+1] += std::string(" ");
      // 
      char_array_size += local_population_rhythm[2*(i-1)].size();
      char_array_size += local_population_rhythm[2*(i-1)+1].size();
      // shift
      population_V_shift_[local_population_] += y[0] + y[4] - y[1] - y[2];
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
  Fijee::Zlib::in_memory_compression( array_to_compress, char_array_size, 
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
Biophysics::Molaee_Ardekani_Wendling_2009::ordinary_differential_equations( double T, const double Y[], double DyDt[] )
{
  // 
  // System of ODE
  // \Phi_{p} -- AMPA
  DyDt[0]  = Y[5];
  DyDt[5]  = A_*a_*sigmoid_P(CPP_*Y[0] - CI1P_*Y[1] - CI2P_*Y[3] + Y[4]) - 2*a_*Y[5] - a_*a_*Y[0];
  //  \Phi_{I} -- GABA_{A,fast}
  DyDt[1]  = Y[6];
  DyDt[6]  = G_*g_*sigmoid_I1(CPI1_*Y[0] - CI1I1_*Y[1] - CI2I1_*Y[2]) - 2*g_*Y[6] - g_*g_*Y[1];
  //  \Phi_{I''} -- GABA_{A,fast}
  DyDt[2]  = Y[7];
  DyDt[7]  = G_*g_*sigmoid_I2(CPI2_*Y[0]               - CI2I2_*Y[3]) - 2*g_*Y[7] - g_*g_*Y[2];
  //  \Phi_{I'} -- GABA_{A,slow}
  DyDt[3]  = Y[8];
  DyDt[8]  = B_*b_*sigmoid_I2(CPI2_*Y[0]               - CI2I2_*Y[3]) - 2*b_*Y[8] - b_*b_*Y[3];
  // P -- AMPA
  DyDt[4]  = Y[9];
  DyDt[9]  = A_*a_*p_[local_population_] - 2*a_*Y[9] - a_*a_*Y[4];

  // 
  // 
  return GSL_SUCCESS;
}
// 
// 
//
void
Biophysics::Molaee_Ardekani_Wendling_2009::modelization_at_electrodes()
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
      for ( int population = 0 ; population < number_samples_ ; population++)
	{
	  // 
	  // 
	  ts_word.clear();
	  ts_values.clear();

	  // 
	  // Inflation of data
	  Fijee::Zlib::in_memory_decompression( population_rhythm_[population], ts_word );
	  // Copy of data
	  int size_of_word = ts_word.size();
	  ts_data = new Bytef[size_of_word];
	  std::copy ( ts_word.begin(), ts_word.end(), ts_data );
	  // 
	  // We lock because of thread collisions (bug C++11?)
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
	      throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
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
  catch( Fijee::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
}
//
//
//
void 
Biophysics::Molaee_Ardekani_Wendling_2009::init()
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
Biophysics::Molaee_Ardekani_Wendling_2009::operator () ( const Pop_to_elec_type Threading )
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
	    throw Fijee::Exit_handler( message, 
				       __LINE__, __FILE__ );
	  };
	}
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }
  catch( Fijee::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
}
