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
// WARNING
// Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
// Afterward, it should be a class member!!
// 
static thread_local int local_electrode_;
// 
// 
// 
Utils::Biophysics::EEG_simulation::EEG_simulation():
  Utils::XML_writer("eeg_alpha_rhythm.xml"),
  number_samples_(0), electrode_(0)
{
}
// 
// 
// 
void
Utils::Biophysics::EEG_simulation::load_files( std::string Output_path, const int Number_of_alpha_files )
{
  // 
  // Load file population's alpha rhythm
  // 

  //
  // Read the populations xml file
  std::cout << "Load populations file for alpha generation" << std::endl;
  // XML output
  set_file_name_( Output_path + "eeg_alpha_rhythm.xml" );

  for( int sub_file = 0 ; sub_file < Number_of_alpha_files ; sub_file++ )
    {
      //
      std::string In_population_file_XML = Output_path + std::string("alpha_rhythm_");
      In_population_file_XML            += std::to_string( sub_file ) + std::string(".xml");
      std::cout << In_population_file_XML << std::endl;
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
	    if( sub_file == 0 )
	      {
		populations_.resize( dipoles_node.attribute("size").as_int() );
		population_rhythm_.resize( dipoles_node.attribute("size").as_int() );
	      }
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
		std::list< std::string > v_time_series;
		int char_array_size = 0;
		// loop over the time series
		for ( auto time_step : dipole )
		  {
		    std::string 
		      ts_time = std::string(time_step.attribute("time").as_string())+std::string(" "),
		      ts_V    = std::string(time_step.attribute("V").as_string())+std::string(" ");
		    // 
		    char_array_size += ts_time.size() + ts_V.size();
		    // 
		    v_time_series.push_back( ts_time );
		    v_time_series.push_back( ts_V );
		  }
		// Check
		if( static_cast<int>(v_time_series.size()) != 2*dipole.attribute("size").as_int() )
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
		// Convert strings into an array of char
		char* array_to_compress = (char*)malloc( char_array_size*sizeof(char) );
		std::list< std::string >::iterator it_ts = v_time_series.begin();
		// first occurence
		strcpy( array_to_compress, (*it_ts).c_str() );
		it_ts++;
		//
		//		while ( it_ts != v_time_series.end() )
		for ( ; it_ts != v_time_series.end() ; it_ts++ )
		  strcat(array_to_compress, (*it_ts).c_str());
		//
		Utils::Zlib::Compression deflate;
		deflate.in_memory_compression( array_to_compress, char_array_size, 
					       population_rhythm_[dipole_number] );
		
		// 
		// Clean area
		delete[] array_to_compress;
		array_to_compress = nullptr;

		// 
		// 
		// 
		//populations_[dipole_number].set_V_time_series( std::move(v_time_series) );
	      }
	    //
	    break;
	  };
	case pugi::status_out_of_memory:
	  {
	    std::cout << "We will use SAX2 libxml2" << std::endl;
	    xml_file.reset();
	    exit(1);
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
  // Load leadfield matrix file
  // 

  //
  // Read the populations xml file
  std::cout << "Load leadfield matrix" << std::endl;
  //
  std::string In_leadfield_matrix_file_XML = Output_path + "../result/leadfield_matrix.xml";
  pugi::xml_document     xml_leadfield_matrix_file;
  pugi::xml_parse_result result_leadfield_matrix = 
    xml_leadfield_matrix_file.load_file( In_leadfield_matrix_file_XML.c_str() );
  //
  switch( result_leadfield_matrix.status )
    {
    case pugi::status_ok:
      {
	//
	// Check that we have a FIJEE XML file
	const pugi::xml_node fijee_node = xml_leadfield_matrix_file.child("fijee");
	if (!fijee_node)
	  {
	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	    exit(1);
	  }

	// 
	// Get sampling
	const pugi::xml_node setup_node = fijee_node.child("setup");
	if (!setup_node)
	  {
	    std::cerr << "Read data from XML: no setup node" << std::endl;
	    exit(1);
	  }
	// 
	std::map< int/*index*/, std::list< std::tuple< std::string /*lavel*/, 
						       int /*dipole*/, 
						       double /*V*/, 
						       double /*I*/ > > > ld_matrix;
	// load the file information in a map
	for ( auto electrodes : setup_node )
	  for ( auto electrode : electrodes )
	    ld_matrix[electrode.attribute("index").as_int()].
	      push_back( std::make_tuple( electrode.attribute("label").as_string(),
					  electrodes.attribute("dipole").as_int(),
					  electrode.attribute("V").as_double(),
					  electrode.attribute("I").as_double() ) );
	
	
	// 
	// Load the map information in the leadfield matrix contenair
	number_samples_ = ld_matrix.size();
	leadfield_matrix_.resize( number_samples_ );
	brain_rhythm_at_electrodes_.resize( number_samples_ );
	// 
	for( auto electrod : ld_matrix )
	  {
	    // 
	    int index_electrode = electrod.first;
	    leadfield_matrix_[index_electrode] =
	      Leadfield_matrix( index_electrode, std::get<0>(*electrod.second.begin()) );
	    // 
	    std::vector< double > v_dipole(electrod.second.size());
	    // 
	    for ( auto v : electrod.second )
	      v_dipole[std::get<1>(v)] = std::get<2>(v);
	    // 
	    leadfield_matrix_[index_electrode].set_V_dipole(std::move(v_dipole));
	  }

	//
	break;
      };
    default:
      {
	std::cerr << "Error reading XML file: " << result_leadfield_matrix.description() 
		  << std::endl;
	exit(1);
      }
    }
}
// 
// 
// 
void
Utils::Biophysics::EEG_simulation::load_tCS_file( std::string Output_path )
{
}
//
//
//
void 
Utils::Biophysics::EEG_simulation::operator () ()
{
  //
  // Mutex the population poping process
  //
  try 
    {
      // lock the population
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      // 
      local_electrode_ = electrode_++;
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }

  // 
  // Generate alpha rhythm for population local_population_
  modelization();
}
// 
// 
//
void
Utils::Biophysics::EEG_simulation::modelization()
{
  // 
  // initialization of vector of vector (must be all the same size)
  brain_rhythm_at_electrodes_[local_electrode_].resize( populations_[0].get_V_time_series_().size() );
  // 
  for( auto time_series : brain_rhythm_at_electrodes_[local_electrode_] )
    time_series = std::make_tuple(0.,0.);
  
  
  // 
  // 
  for( int dipole = 0 ; dipole < static_cast<int>(populations_.size()) ; dipole++ )
    {
      int time_step = 0;
      for( auto time_series : populations_[dipole].get_V_time_series_() )
	{
	  // 
	  // Partial check
	  if ( std::get<0>(brain_rhythm_at_electrodes_[local_electrode_][time_step]) == 0 )
	    {
	      std::get<0>(brain_rhythm_at_electrodes_[local_electrode_][time_step]) = 
		std::get<0>(time_series);
	      //
	      std::get<1>(brain_rhythm_at_electrodes_[local_electrode_][time_step++]) += 
		(leadfield_matrix_[local_electrode_].get_V_dipole_())[dipole]*std::get<1>(time_series);
	    }
	  else if( std::get<0>(brain_rhythm_at_electrodes_[local_electrode_][time_step]) == std::get<0>(time_series) )
	    {
	      std::get<1>(brain_rhythm_at_electrodes_[local_electrode_][time_step++]) += 
		(leadfield_matrix_[local_electrode_].get_V_dipole_())[dipole]*std::get<1>(time_series);
	    }
	  else
	    {
	      std::cerr << "Problem on the time step: "
			<< "time step read: " << std::get<0>(time_series)
			<< "s, and it should be: " << std::get<0>(brain_rhythm_at_electrodes_[local_electrode_][time_step])
			<< "s."
			<< std::endl;
	      abort();
	    }
	}
    }
}
// 
// 
// 
void
Utils::Biophysics::EEG_simulation::Make_analysis()
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
  for ( auto electrode : leadfield_matrix_)
    output_stream_ <<  electrode.get_label_() << " ";
  // 
  output_stream_ << std::endl;

  // 
  // R values: alpha rhythm
  // 
  std::vector< std::vector< std::tuple< double, double > >::const_iterator > 
    it( number_samples_ );
  // 
  for( int electrode = 0 ; electrode < number_samples_ ; electrode++ )
    it[electrode] = brain_rhythm_at_electrodes_[electrode].begin();
  // 
  while( it[0] !=  brain_rhythm_at_electrodes_[0].end())
    {
      // get the time
      output_stream_ <<  std::get<0>( *(it[0]) ) << " ";
      // 
     for (int electrode = 0 ; electrode < number_samples_ ; electrode++)
       output_stream_ <<  std::get<1>( *(it[electrode]++) )  << " ";
      // 
      output_stream_ << std::endl;
    }

  //
  //
  Make_output_file("eeg_alpha_rhythm.frame");


  // 
  // FFT
  // 

  //
  // R header: Power spectral density
  // 
  output_stream_
    << "Hz ";
  // 
  for ( auto electrode : leadfield_matrix_)
    output_stream_ <<  electrode.get_label_() << " ";
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
  for ( int electrode = 0 ; electrode < number_samples_ ; electrode++ )
    {
      n = 0;
      data_vector[electrode] = new double[2*2048/*N*/];
      for( auto time_potential : brain_rhythm_at_electrodes_[electrode] )
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
    gsl_fft_complex_radix2_forward (electrode, 1/*stride*/, 2048);

  // 
  // Average the power signal
  std::vector< double > average_power(N);
 
  // 
  for ( int i = 0 ; i < N ; i++ )
    {
      // 
      for ( auto electrode : data_vector )
	{
	  average_power[i]  = REAL(electrode,i)*REAL(electrode,i);
	  average_power[i] += IMAG(electrode,i)*IMAG(electrode,i);
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
      for ( auto electrode : data_vector )
	output_stream_ 
	  << (REAL(electrode,i)*REAL(electrode,i) + IMAG(electrode,i)*IMAG(electrode,i)) / N
	  << " ";
	  
      // 
      output_stream_ << average_power[i] << std::endl;
    }

  //
  //
  Make_output_file("eeg_PSD.frame");
#endif
#endif      
}
// 
// 
// 
void 
Utils::Biophysics::EEG_simulation::output_XML()
{
  // 
  // Build XML output 
  // 
  
  // 
  // Output XML file initialization
  auto setup_node = fijee_.append_child("setup");
  setup_node.append_attribute("size") = static_cast<int>(brain_rhythm_at_electrodes_[0].size());

  // WARNING
  // NOTA: XML are too huge, adviced to produice only R project output (-DTRACE=100)
//  // 
//  // 
//  std::vector< std::vector< std::tuple< double, double > >::const_iterator > 
//    it( number_samples_ );
//  // 
//  for( int electrode = 0 ; electrode < number_samples_ ; electrode++ )
//    it[electrode] = brain_rhythm_at_electrodes_[electrode].begin();
//  
//  // 
//  // 
//  int index = 0;
//  // 
//  while( it[0] != brain_rhythm_at_electrodes_[0].end() )
//    {
//      auto electrodes_node = setup_node.append_child("electrodes");
//      electrodes_node.append_attribute("index")  = index++;
//      electrodes_node.append_attribute("dipole") = 0;
//      electrodes_node.append_attribute("time")   = std::get<0>(*it[0]);
//      electrodes_node.append_attribute("size")   = number_samples_;
// 
//      // 
//      for( int electrode = 0 ; electrode < number_samples_ ; electrode++ )
//	{
//	  // 
//	  // 
//	  auto electrode_node = electrodes_node.append_child("electrode");
//	  // 
//	  electrode_node.append_attribute("index")  = electrode;
//	  electrode_node.append_attribute("label")  = leadfield_matrix_[electrode].get_label_().c_str();
//	  electrode_node.append_attribute("V")  = std::get<1>(*it[electrode]);
//	  electrode_node.append_attribute("I")  = 0.;
//	  // 
//	}
//    }

  // 
  // Statistical analysise
  Make_analysis();
}
