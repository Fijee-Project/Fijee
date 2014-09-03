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
#include "Brain_rhythm.h"
// 
// 
// 
Utils::Biophysics::Brain_rhythm::Brain_rhythm( const int Duration ):
  Utils::XML_writer("alpha_rhythm.xml"),
  duration_(Duration)
{}
// 
// 
// 
void
Utils::Biophysics::Brain_rhythm::load_population_file( std::string Output_path )
{
  //
  // Read the populations xml file
  std::cout << "Load populations file for alpha generation" << std::endl;

  // 
  // Load file
  std::string In_population_file_XML = Output_path + "dipoles.xml";
  // XML output
  set_file_name_( Output_path + "alpha_rhythm.xml" );

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
	number_samples_ = dipoles_node.attribute("size").as_int();
	populations_.resize(number_samples_);
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
	    //
	    population_rhythm_.resize(number_samples_);
	    population_V_shift_.resize(number_samples_);
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
Utils::Biophysics::Brain_rhythm::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
  // 
  // 

  //
  // R header: alpha rhythm
  // 
  output_stream_
    << "time ";
  // 
  for (int population = 0 ; population < number_samples_ ; population++)
    output_stream_ <<  population << " ";
  // 
  output_stream_ << std::endl;

  // 
  // 
  try{
    // 
    // R values: alpha rhythm
    // 

    // 
    // Inflation of data
    std::vector< std::vector< Bytef > > ts_word(number_samples_);
    std::vector< std::vector< std::string > > ts_values(number_samples_);
    //
    char*  pch;
    Bytef* ts_data;
    // 
    Utils::Zlib::Compression deflate;
    // 
    for (int population = 0 ; population < number_samples_ ; population++)
      {
	deflate.in_memory_decompression( population_rhythm_[population], ts_word[population] );
	// Copy of data
	int size_of_word = ts_word[population].size();
	ts_data = (Bytef*) malloc( size_of_word * sizeof(Bytef) );
	std::copy ( ts_word[population].begin(), ts_word[population].end(), ts_data );
	// Cut of data
	pch = strtok(reinterpret_cast<char*>(ts_data)," ");
	//
	while ( pch != nullptr && size_of_word != 0 )
	  {
	    size_of_word -= strlen(pch) + 1;
	    ts_values[population].push_back( std::string(pch) );
	    pch = strtok (nullptr, " ");
	  }
	// 
	if( static_cast<int>(ts_values[population].size()) != 2*(duration_-1) )
	  {
	    std::string message = std::string("The size of the inflated data structure is: ")
	      + std::to_string(ts_values[population].size()) 
	      + std::string(". It should be: ") + std::to_string(2*(duration_-1))
	      + std::string(".");
	    //
	    throw Utils::Error_handler( message,  __LINE__, __FILE__ );
	  }
	
	// 
	//
	delete[] ts_data;
	ts_data = nullptr;
	// 
	pch = nullptr;
     }


    // 
    // Loop over the time series
    for( int i = 1 ; i < duration_ ; i++ )
      {
	// get the time
	output_stream_ <<  ts_values[0][2*(i-1)] << " ";
	// get the potential
	for (int population = 0 ; population < number_samples_ ; population++)
	  {
	    // 
	    double V = std::stod( ts_values[population][2*(i-1) + 1] );
	    V -= population_V_shift_[population];
	    // 
	    output_stream_  << V << " ";
	  }
	// End the stream
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
    for (int population = 0 ; population < number_samples_ ; population++)
      output_stream_ <<  population << " ";
    // 
    output_stream_ << "power" << std::endl;


    // 
    // R values: Power spectral density
    // 
    int 
      N = 2048, /* N is a power of 2 */
      n = 0;
    // real and imaginary
    std::vector< double* > data_vector(number_samples_);
    // 
    for ( int population = 0 ; population < number_samples_ ; population++ )
      {
	n = 0;
	data_vector[population] = new double[2*2048/*N*/];
	for( int i = 1 ; i < duration_ ; i++ )
	  {
	    if ( n < N )
	      {
		// 
		double V = std::stod(std::string(ts_values[population][2*(i-1) + 1]));
		//
		REAL(data_vector[population],n) = V;
		IMAG(data_vector[population],n++) = 0.0;
	      }
	    else
	      break;
	  }
      }

    // 
    // Forward FFT
    // A stride of 1 accesses the array without any additional spacing between elements. 
    for ( auto population : data_vector )
      gsl_fft_complex_radix2_forward (population, 1/*stride*/, 2048);

    // 
    // Average the power signal
    std::vector< double > average_power(N);
 
    // 
    for ( int i = 0 ; i < N ; i++ )
      {
	// 
	for ( auto population : data_vector )
	  {
	    average_power[i]  = REAL(population,i)*REAL(population,i);
	    average_power[i] += IMAG(population,i)*IMAG(population,i);
	    average_power[i] /= N;
	  }
	// 
	average_power[i] /= number_samples_;
      }
 

    //
    // 
    for ( int i = 0 ; i < N ; i++ )
      {
	output_stream_ 
	  << i << " ";
	// 
	for ( auto population : data_vector )
	  output_stream_ 
	    << (REAL(population,i)*REAL(population,i) + IMAG(population,i)*IMAG(population,i)) / N
	    << " ";
	  
	// 
	output_stream_ << average_power[i] << std::endl;
      }

    //
    //
    Make_output_file("PSD.frame");
  }
  catch( Utils::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
#endif
#endif      
}
// 
// 
// 
void 
Utils::Biophysics::Brain_rhythm::output_XML()
{
  // 
  // Statistical analysise
  Make_analysis();

  //
  //
  try
    {  // 
      // Build XML output 
      // 
  
      // 
      // Output XML file initialization
      dipoles_node_ = fijee_.append_child("dipoles");
      dipoles_node_.append_attribute("size") = static_cast<int>( populations_.size() );

      // 
      // loop over the time series
      for( int population = 0 ; population < number_samples_ ; population++ )
	{
	  // 
	  // 
	  dipole_node_ = dipoles_node_.append_child("dipole");
	  // 
	  dipole_node_.append_attribute("index")  = populations_[population].get_index_();
	  // 
	  dipole_node_.append_attribute("x") = (populations_[population].get_position_())[0];
	  dipole_node_.append_attribute("y") = (populations_[population].get_position_())[1];
	  dipole_node_.append_attribute("z") = (populations_[population].get_position_())[2];
	  // 
	  dipole_node_.append_attribute("vx") = (populations_[population].get_direction_())[0];
	  dipole_node_.append_attribute("vy") = (populations_[population].get_direction_())[1];
	  dipole_node_.append_attribute("vz") = (populations_[population].get_direction_())[2];
	  // 
	  dipole_node_.append_attribute("I") = populations_[population].get_I_();
	  // 
	  dipole_node_.append_attribute("index_cell") = populations_[population].get_index_cell_();
	  dipole_node_.append_attribute("index_parcel") = populations_[population].get_index_parcel_();
	  // 
	  dipole_node_.append_attribute("lambda1") = (populations_[population].get_lambda_())[0];
	  dipole_node_.append_attribute("lambda2") = (populations_[population].get_lambda_())[1];
	  dipole_node_.append_attribute("lambda3") = (populations_[population].get_lambda_())[2];
	  //
	  dipole_node_.append_attribute("size") = static_cast<int>( duration_ - 1 );

	  // 
	  // Time series
	  // 

	  // 
	  // Inflation of data
	  std::vector< std::vector< Bytef > > ts_word(number_samples_);
	  std::vector< std::vector< std::string > > ts_values(number_samples_);
	  //
	  char*  pch;
	  Bytef* ts_data;
	  // 
	  Utils::Zlib::Compression deflate;
	  deflate.in_memory_decompression( population_rhythm_[population], ts_word[population] );
	  // Copy of data
	  int size_of_word = ts_word[population].size();
	  ts_data = (Bytef*) malloc( size_of_word * sizeof(Bytef) );
	  std::copy ( ts_word[population].begin(), ts_word[population].end(), ts_data );
	  //  Cut of data
	  pch = strtok(reinterpret_cast<char*>(ts_data)," ");
	  //
	  while ( pch != nullptr && size_of_word != 0 )
	    {
	      size_of_word -= strlen(pch) + 1;
	      ts_values[population].push_back(std::string(pch));
	      pch = strtok (nullptr, " ");
	    }
	  // 
	  if( static_cast<int>(ts_values[population].size()) != 2*(duration_-1) )
	    {
	      std::string message = std::string("The size of the inflated data structure is: ")
		+ std::to_string(ts_values[population].size()) 
		+ std::string(". It should be: ") + std::to_string(2*(duration_-1))
		+ std::string(".");
	      //
	      throw Utils::Error_handler( message,  __LINE__, __FILE__ );
	    }

	  // 
	  // Loop over the time series
	  int index_time_step = 0;
	  //      while( it[population] !=  population_rhythm_[population].end() )
	  for( int i = 1 ; i < duration_ ; i++ )
	    {
	      //       
	      time_series_node_ = dipole_node_.append_child("time_step");
	      // 
	      time_series_node_.append_attribute("index") = index_time_step++;
	      time_series_node_.append_attribute("time")  = ts_values[population][2*(i-1)].c_str();
	      // conversion mV -> V
	      double V  = std::stod(ts_values[population][2*(i-1) + 1]);
	      V        -= population_V_shift_[population];
	      // 
	      time_series_node_.append_attribute("V")     = V * 1.e-03;
	    }

	  //
	  // clear the data
	  delete[] ts_data;
	  ts_data = nullptr;
//	  // 
//	  delete[] pch;
	  pch = nullptr;
	  //
	  population_rhythm_[population].clear();
	}
    }
  catch( Utils::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
}
