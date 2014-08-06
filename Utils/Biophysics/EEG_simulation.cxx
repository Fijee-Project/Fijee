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
#include "EEG_simulation.h"
// 
// 
// 
Utils::Biophysics::EEG_simulation::EEG_simulation():
  Utils::XML_writer("eeg_alpha_rhythm.xml")
{
}
// 
// 
// 
Utils::Biophysics::EEG_simulation::EEG_simulation( std::string Output_path ):
  Utils::XML_writer("eeg_alpha_rhythm.xml")
{
  //
  // Read the populations xml file
  std::cout << "Load populations file for alpha generation" << std::endl;

  // 
  // Load file
  std::string In_population_file_XML = Output_path + "alpha_rhythm.xml";

  //
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file( In_population_file_XML.c_str() );
  //
  switch( result.status )
    {
    case pugi::status_ok:
      {
	//
	// Check that we have a FIJEE XML file
	const pugi::xml_node fijee_node = xml_file.child("fijee");
	if (!fijee_node)
	  {
	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	    exit(1);
	  }

	// 
	// Get sampling
	const pugi::xml_node dipoles_node = fijee_node.child("dipoles");
	if (!dipoles_node)
	  {
	    std::cerr << "Read data from XML: no dipoles node" << std::endl;
	    exit(1);
	  }
	// Get the number of samples
	populations_.resize( dipoles_node.attribute("size").as_int() );
	// loop over the samples
	for ( auto dipole : dipoles_node )
	  {
	    //
	    // Get the number of populations
	    int dipole_number = dipole.attribute("index").as_int();

	    // 
	    // load dipole information in populations vector
	    Population neurons( dipole_number,
				dipole.attribute("x").as_double(),
				dipole.attribute("y").as_double(),
				dipole.attribute("z").as_double(),
				dipole.attribute("vx").as_double(),
				dipole.attribute("vy").as_double(),
				dipole.attribute("vz").as_double(),
				dipole.attribute("I").as_double(),
				dipole.attribute("index_cell").as_int(),
				dipole.attribute("index_parcel").as_int(),
				dipole.attribute("lambda1").as_double(),
				dipole.attribute("lambda2").as_double(),
				dipole.attribute("lambda3").as_double() );
	    // 
	    populations_[dipole_number] = std::move( neurons );

	    // 
	    // Get time step sampling
	    std::list< std::tuple<double,double> > v_time_series;
	    // loop over the time series
	    for ( auto time_step : dipole )
	      v_time_series.push_back( std::make_tuple( time_step.attribute("time").as_double(),
							time_step.attribute("V").as_double() ));
	    // Check
	    if( v_time_series.size() != dipole.attribute("size").as_int() )
	      {
		std::cerr << "The potential time series list size does not match the "
			  << In_population_file_XML
			  << " file."
			  << std::endl;
		std::cerr << "Potential time series list size: " << v_time_series.size() 
			  << std::endl;
		std::cerr << In_population_file_XML
			  << " file: " <<  dipole.attribute("size").as_int()
			  << std::endl;
		abort();
	      }
	    // 
	    populations_[dipole_number].set_V_time_series( std::move(v_time_series) );
	  }
	//
	break;
      };
    default:
      {
	std::cerr << "Error reading XML file: " << result.description() << std::endl;
	exit(1);
      }
    }
}
// 
// 
// 
void
Utils::Biophysics::EEG_simulation::load_population_file( std::string Output_path )
{
//  //
//  // Read the populations xml file
//  std::cout << "Load populations file for alpha generation" << std::endl;
//
//  // 
//  // Load file
//  std::string In_population_file_XML = Output_path + "parcellation.xml";
//  // XML output
//  set_file_name_( Output_path + "alpha_rhythm.xml" );
//
//  //
//  pugi::xml_document     xml_file;
//  pugi::xml_parse_result result = xml_file.load_file( In_population_file_XML.c_str() );
//  //
//  switch( result.status )
//    {
//    case pugi::status_ok:
//      {
//	//
//	// Check that we have a FIJEE XML file
//	const pugi::xml_node fijee_node = xml_file.child("fijee");
//	if (!fijee_node)
//	  {
//	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
//	    exit(1);
//	  }
//
//	// 
//	// Get sampling
//	const pugi::xml_node dipoles_node = fijee_node.child("dipoles");
//	if (!dipoles_node)
//	  {
//	    std::cerr << "Read data from XML: no dipoles node" << std::endl;
//	    exit(1);
//	  }
//	// Get the number of samples
//	number_samples_ = dipoles_node.attribute("size").as_int();
//	populations_.resize(number_samples_);
//	// loop over the samples
//	for ( auto dipole : dipoles_node )
//	  {
//	    //
//	    // Get the number of populations
//	    int dipole_number = dipole.attribute("index").as_int();
//
//	    // 
//	    // load dipole information
//	    // Position
//	    populations_[dipole_number].position_[0] = dipole.attribute("x").as_double();
//	    populations_[dipole_number].position_[1] = dipole.attribute("y").as_double();
//	    populations_[dipole_number].position_[2] = dipole.attribute("z").as_double();
//	    // Direction
//	    populations_[dipole_number].direction_[0] = dipole.attribute("vx").as_double();
//	    populations_[dipole_number].direction_[1] = dipole.attribute("vy").as_double();
//	    populations_[dipole_number].direction_[2] = dipole.attribute("vz").as_double();
//	    // Current and potential
//	    populations_[dipole_number].I_ = dipole.attribute("I").as_double();
//	    // cell index and parcel
//	    populations_[dipole_number].index_cell_ = dipole.attribute("index_cell").as_int();
//	    populations_[dipole_number].parcel_     = 0;//dipole.attribute("").as_int();
//	    // Lambda
//	    populations_[dipole_number].lambda_[0] = dipole.attribute("lambda1").as_double();
//	    populations_[dipole_number].lambda_[1] = dipole.attribute("lambda2").as_double();
//	    populations_[dipole_number].lambda_[2] = dipole.attribute("lambda3").as_double();
//	    
//
//	    //
//	    //
//	    population_rhythm_.resize(number_samples_);
//	    population_V_shift_.resize(number_samples_);
//	  }
//	//
//	break;
//      };
//    default:
//      {
//	std::cerr << "Error reading XML file: " << result.description() << std::endl;
//	exit(1);
//      }
//    }
}
// 
// 
// 
void
Utils::Biophysics::EEG_simulation::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
//  // 
//  // 
//
//  //
//  // R header: alpha rhythm
//  // 
//  output_stream_
//    << "time ";
//  // 
//  for (int population = 0 ; population < number_samples_ ; population++)
//    output_stream_ <<  population << " ";
//  // 
//  output_stream_ << std::endl;
//
//  // 
//  // R values: alpha rhythm
//  // 
//  std::vector< std::list< std::tuple< double, double > >::const_iterator > 
//    it(number_samples_);
//  // 
//  for( int population = 0 ; population < number_samples_ ; population++ )
//    it[population] = population_rhythm_[population].begin();
//  // 
//  while( it[0] !=  population_rhythm_[0].end())
//    {
//      // get the time
//      output_stream_ <<  std::get<0>( *(it[0]) ) << " ";
//      // 
//     for (int population = 0 ; population < number_samples_ ; population++)
//       output_stream_ <<  std::get<1>( *(it[population]++) ) - population_V_shift_[population] << " ";
//      // 
//      output_stream_ << std::endl;
//    }
//
//  //
//  //
//  Make_output_file("alpha_rhythm.frame");
//
//
//  // 
//  // FFT
//  // 
//
//  //
//  // R header: Power spectral density
//  // 
//  output_stream_
//    << "Hz ";
//  // 
//  for (int population = 0 ; population < number_samples_ ; population++)
//    output_stream_ <<  population << " ";
//  // 
//  output_stream_ << "power" << std::endl;
//
//
//  // 
//  // R values: Power spectral density
//  // 
//  int 
//    N = 2048, /* N is a power of 2 */
//    n = 0;
//  // real and imaginary
//  std::vector< double* > data_vector(number_samples_);
//  // 
//  for ( int population = 0 ; population < number_samples_ ; population++ )
//    {
//      n = 0;
//      data_vector[population] = new double[2*2048/*N*/];
//      for( auto time_potential : population_rhythm_[population] )
//	{
//	  if ( n < N )
//	    {
//	      REAL(data_vector[population],n)   = std::get<1>(time_potential);
//	      IMAG(data_vector[population],n++) = 0.0;
//	    }
//	  else
//	    break;
//	}
//    }
//
//  // 
//  // Forward FFT
//  // A stride of 1 accesses the array without any additional spacing between elements. 
//  for ( auto population : data_vector )
//    gsl_fft_complex_radix2_forward (population, 1/*stride*/, 2048);
//
//  // 
//  // Average the power signal
//  std::vector< double > average_power(N);
// 
//  // 
//  for ( int i = 0 ; i < N ; i++ )
//    {
//      // 
//      for ( auto population : data_vector )
//	{
//	  average_power[i]  = REAL(population,i)*REAL(population,i);
//	  average_power[i] += IMAG(population,i)*IMAG(population,i);
//	  average_power[i] /= N;
//	}
//      // 
//      average_power[i] /= number_samples_;
//    }
// 
//
//  //
//  // 
//  for ( int i = 0 ; i < N ; i++ )
//    {
//      output_stream_ 
//	<< i << " ";
//      // 
//      for ( auto population : data_vector )
//	output_stream_ 
//	  << (REAL(population,i)*REAL(population,i) + IMAG(population,i)*IMAG(population,i)) / N
//	  << " ";
//	  
//      // 
//      output_stream_ << average_power[i] << std::endl;
//    }
//
//  //
//  //
//  Make_output_file("PSD.frame");
#endif
#endif      
}
// 
// 
// 
void 
Utils::Biophysics::EEG_simulation::output_XML()
{
//  // 
//  // Build XML output 
//  // 
//  
//  // 
//  // Output XML file initialization
//  dipoles_node_ = fijee_.append_child("dipoles");
//  dipoles_node_.append_attribute("size") = static_cast<int>( populations_.size() );
//  
//  // 
//  std::vector< std::list< std::tuple< double, double > >::const_iterator > 
//    it(number_samples_);
//  
//  // 
//  // loop over the time series
//  int index = 0;
//  // 
//  for( int population = 0 ; population < number_samples_ ; population++ )
//    {
//      // 
//      // 
//      dipole_node_ = dipoles_node_.append_child("dipole");
//      // 
//      dipole_node_.append_attribute("index")  = index++;
//      // 
//      dipole_node_.append_attribute("x") = populations_[population].position_[0];
//      dipole_node_.append_attribute("y") = populations_[population].position_[1];
//      dipole_node_.append_attribute("z") = populations_[population].position_[2];
//      // 
//      dipole_node_.append_attribute("vx") = populations_[population].direction_[0];
//      dipole_node_.append_attribute("vy") = populations_[population].direction_[1];
//      dipole_node_.append_attribute("vz") = populations_[population].direction_[2];
//      // 
//      dipole_node_.append_attribute("I") = populations_[population].I_;
//      // 
//      dipole_node_.append_attribute("index_cell") = populations_[population].index_cell_;
//      dipole_node_.append_attribute("parcel") = populations_[population].parcel_;
//      // 
//      dipole_node_.append_attribute("lambda1") = populations_[population].lambda_[0];
//      dipole_node_.append_attribute("lambda2") = populations_[population].lambda_[1];
//      dipole_node_.append_attribute("lambda3") = populations_[population].lambda_[2];
//      //
//      dipole_node_.append_attribute("size") = static_cast<int>( population_rhythm_[0].size() );
//
//      // 
//      // Time series
//      it[population] = population_rhythm_[population].begin();
//      // 
//      int index_time_step = 0;
//      while( it[population] !=  population_rhythm_[population].end() )
//	{
//	  //       
//	  time_series_node_ = dipole_node_.append_child("time_step");
//	  // 
//	  time_series_node_.append_attribute("index") = index_time_step++;
//	  time_series_node_.append_attribute("time")  = std::get<0>( *(it[population]) );
//	  // conversion mV -> V
//	  double V = std::get<1>( *(it[population]++) ) - population_V_shift_[population];
//	  time_series_node_.append_attribute("V")     = V * 1.e-03;
//	}
//    }
//
//  
//  // 
//  // Statistical analysise
//  Make_analysis();
}
