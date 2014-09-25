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
Biophysics::Brain_rhythm::Brain_rhythm( const int Duration ):
  Fijee::XML_writer( "alpha_rhythm.xml", /*heavy XML*/ true ),
  duration_(Duration), number_samples_(0), number_electrodes_(0), number_parcels_(0),
  electrode_(0)
{}
// 
// 
// 
void
Biophysics::Brain_rhythm::load_population_file( std::string Output_path )
{
  //
  // Read the populations xml file
  std::cout << "Load populations file for alpha generation" << std::endl;

  // 
  // Load file
  std::string In_population_file_XML = Output_path + "parcellation.xml";
  // XML output
  alpha_rhythm_output_file_ = Output_path + std::string("alpha_rhythm.xml");
  set_file_name_( Output_path + "alpha_rhythm.xml" );
  //  out_XML_file_ = fopen( alpha_rhythm_output_file_.c_str(), "w" );

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
Biophysics::Brain_rhythm::load_leadfield_matrix_file( std::string Output_path )
{

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
	number_electrodes_ = ld_matrix.size();
	leadfield_matrix_.resize( number_electrodes_ );
	brain_rhythm_at_electrodes_.resize( number_electrodes_ );
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
Biophysics::Brain_rhythm::load_electric_field_file( std::string Output_path, const double Lambda )
{

  // 
  // Load electric field file
  // 
  
  //
  // Read the electric field xml file
  std::cout << "Load electric field from parcellation" << std::endl;
  //
  std::string In_electric_field_file_XML = Output_path + "../result/Electric_field_parcellation.xml";
  pugi::xml_document xml_electric_field_file;
  pugi::xml_parse_result result_electric_field = 
    xml_electric_field_file.load_file( In_electric_field_file_XML.c_str() );
  //
  switch( result_electric_field.status )
    {
    case pugi::status_ok:
      {
	//
	// Check that we have a FIJEE XML file
	const pugi::xml_node fijee_node = xml_electric_field_file.child("fijee");
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
	// 
	// Get the number of samples
	number_parcels_ = dipoles_node.attribute("size").as_int();
	parcellation_.resize(number_parcels_);
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
	    parcellation_[dipole_number] = std::move( neurons );
	    
	    // 
	    // Electric field value
	    std::list< std::tuple< double, double > > ts;
	    // 
	    for( auto physical_value : dipole )
	      {
		// 
		// time
		double time = physical_value.attribute("time").as_double();
		
		// 
		// Dot product between field and the normal of the parcel centroid
		double field_E[3] = {
		  physical_value.attribute("val_0").as_double(),
		  physical_value.attribute("val_1").as_double(),
		  physical_value.attribute("val_2").as_double() 
		};
		//
		double V_tCS = field_E[0]*(parcellation_[dipole_number].get_direction_())[0];
		V_tCS       += field_E[1]*(parcellation_[dipole_number].get_direction_())[1];
		V_tCS       += field_E[2]*(parcellation_[dipole_number].get_direction_())[2];
		V_tCS       *= Lambda; // scaling factor
		
		// 
		// 
		ts.push_back( std::make_tuple(time, V_tCS) );
	      }
	    
	    
	    // 
	    // record time series values
	    parcellation_[dipole_number].set_V_time_series( std::move(ts) );
	  }
	
	//
	break;
      };
    default:
      {
	std::cerr << "Error reading XML file: " << result_electric_field.description() 
		  << std::endl;
	exit(1);
      }
    }
}
// 
// 
// 
void
Biophysics::Brain_rhythm::Make_analysis()
{
#ifdef FIJEE_TRACE
#if FIJEE_TRACE == 100
  // 
  // 

  //
  // R header: alpha rhythm
  // 
  int Max_samples_studied = ( number_samples_ > 100 ? 100 : number_samples_ );
  //
  std::cout << "Alpha rhythm analysis file" << std::endl;
  output_stream_
    << "time ";
  // 
  for (int population = 0 ; 
       population <  Max_samples_studied;
       population++)
    output_stream_ <<  population << " ";
  // 
  output_stream_ << std::endl;
  // 
  Make_output_file("alpha_rhythm.frame");

  // 
  // 
  try{
    // 
    // R values: alpha rhythm
    // 

    // 
    // Inflation of data
    std::vector< std::vector< Bytef > > ts_word(Max_samples_studied);
    std::vector< std::vector< std::string > > ts_values(Max_samples_studied);
    //
    char*  pch;
    Bytef* ts_data;
    // 
    for ( int population = 0 ; population < Max_samples_studied ; population++)
      {
	Fijee::Zlib::in_memory_decompression( population_rhythm_[population], ts_word[population] );
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
	    throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
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
	for ( int population = 0 ; population < Max_samples_studied ; population++)
	  {
	    // 
	    double V = std::stod( ts_values[population][2*(i-1) + 1] );
	    V -= population_V_shift_[population];
	    // 
	    output_stream_  << V << " ";
	  }
	// End the stream
	output_stream_ << std::endl;

	// 
	// Flush file in the mid time
	if( i % 100 == 0 || i == duration_ - 1 )
	  Append_output_file("alpha_rhythm.frame");
      }


    // 
    // FFT
    // 

    //
    // R header: Power spectral density
    // 
    std::cout << "Power spectral density analysis file" << std::endl;
    output_stream_
      << "Hz ";
    // 
    for (int population = 0 ; population < Max_samples_studied ; population++)
      output_stream_ <<  population << " ";
    // 
    output_stream_ << "power" << std::endl;
    // 
    Make_output_file("PSD.frame");

    // 
    // R values: Power spectral density
    // 
    int 
      N = 2048, /* N is a power of 2 */
      n = 0;
    // real and imaginary
    std::vector< double* > data_vector( Max_samples_studied );
    // 
    for ( int population = 0 ; population < Max_samples_studied ; population++ )
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
	average_power[i] /= static_cast<double>(Max_samples_studied);
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

	// 
	// 
	if( ( i % 100 == 0 && i > 1 ) || i == N - 1 )
	  Append_output_file("PSD.frame");
      }


    //
    // Alpha rhythm at electrodes
    // 
    
    if( !brain_rhythm_at_electrodes_.empty() )
      {
	// 
	// 
	std::cout << "Alpha rhythm at electrodes analysis file" << std::endl;
	
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
	for ( int i = 1 ; i < duration_ ; i++ )
	  {
	    // 
	    // time 
	    output_stream_ << brain_rhythm_at_electrodes_[0][2*(i-1)] << " ";
	    
	    // 
	    // Potential at electrodes
	    for( int electrode = 0 ; electrode < number_electrodes_ ; electrode++ )
	      output_stream_ << brain_rhythm_at_electrodes_[electrode][2*(i-1)+1] << " ";
	    
	    //
	    // 
	    output_stream_ << std::endl;
	  }
	
	// 
	//
	Make_output_file("alpha_rhythm_electrodes.frame");
	
	// 
	// FFT
	// 
	
	//
	// R header: Power spectral density
	// 
	std::cout << "Power spectral density at electrodes analysis file" << std::endl;
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
	std::vector< double* > data_vector( number_electrodes_ );
	// 
	for ( int electrode = 0 ; electrode < number_electrodes_ ; electrode++ )
	  {
	    n = 0;
	    data_vector[electrode] = new double[2*2048/*N*/];
	    for( int i = 1 ; i < duration_ ; i++ )
	      {
		if ( n < N )
		  {
		    // 
		    double V = brain_rhythm_at_electrodes_[electrode][2*(i-1) + 1];
		    //
		    REAL(data_vector[electrode],n) = V;
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
	    average_power[i] /= static_cast<double>(number_electrodes_);
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
	Make_output_file("PSD_electrodes.frame");
      }
  }
  catch( Fijee::Exception_handler& err )
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
Biophysics::Brain_rhythm::output_XML()
{
  // 
  // Statistical analysise
  Make_analysis();

  //
  //
  std::cout << "XML output files" << std::endl;
  //
  try
    {  

      // 
      // A lot of populations of neurones overload the RAM. We process chunk by chunk to output the 
      // XML.
      // Thanks to Arseny Kapoulkine: 
      // The approach that Arseny usually recommends in this case is as follows:
      // 
      //   1. Manually write the XML declaration and the root tag to the stream (<root>)
      //   2. For each chunk of data, repeat:
      //   2.1. Append all necessary children to the document
      //   2.2. Save them to the stream - you can use xml_node::output/xml_node::print for this
      //   2.3. Remove all children from the document, or just reset() the document.
      //   3. Write the closing root tag to the stream (</root>)
      // 
      //      std::string text("<?xml version=\"1.0\"?>\n");
      //      text += std::string("<fijee xmlns:fijee=\"https://github.com/Fijee-Project/Fijee\">\n");
      //      // 
      //      fwrite ( text.c_str(), sizeof(char), static_cast<int>( text.size() ), out_XML_file_ );
      //      // 
      //      pugi::xml_writer_file writer_file( out_XML_file_ );

      int file_number = 0;

      // 
      // Build XML output 
      // 
  
      // 
      // Output XML file initialization
      pugi::xml_node dipoles_node = fijee_.append_child("dipoles");
      dipoles_node.append_attribute("size") = static_cast<int>( populations_.size() );

      // 
      // loop over the time series
      for( int population = 0 ; population < number_samples_ ; population++ )
	{
	  pugi::xml_node dipole_node = dipoles_node.append_child("dipole");
	  // 
	  dipole_node.append_attribute("index")  = populations_[population].get_index_();
	  // 
	  dipole_node.append_attribute("x") = (populations_[population].get_position_())[0];
	  dipole_node.append_attribute("y") = (populations_[population].get_position_())[1];
	  dipole_node.append_attribute("z") = (populations_[population].get_position_())[2];
	  // 
	  dipole_node.append_attribute("vx") = (populations_[population].get_direction_())[0];
	  dipole_node.append_attribute("vy") = (populations_[population].get_direction_())[1];
	  dipole_node.append_attribute("vz") = (populations_[population].get_direction_())[2];
	  // 
	  dipole_node.append_attribute("I") = populations_[population].get_I_();
	  // 
	  dipole_node.append_attribute("index_cell") = populations_[population].get_index_cell_();
	  dipole_node.append_attribute("index_parcel") = populations_[population].get_index_parcel_();
	  // 
	  dipole_node.append_attribute("lambda1") = (populations_[population].get_lambda_())[0];
	  dipole_node.append_attribute("lambda2") = (populations_[population].get_lambda_())[1];
	  dipole_node.append_attribute("lambda3") = (populations_[population].get_lambda_())[2];
	  //
	  dipole_node.append_attribute("size") = static_cast<int>( duration_ - 1 );

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
	  Fijee::Zlib::in_memory_decompression( population_rhythm_[population], ts_word[population] );
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
	      throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	    }


	  // 
	  // Loop over the time series
	  int index_time_step = 0;
	  //
	  for( int i = 1 ; i < duration_ ; i++ )
	    {
	      //       
	      pugi::xml_node time_series_node = dipole_node.append_child("time_step");
	      // 
	      time_series_node.append_attribute("index") = index_time_step++;
	      time_series_node.append_attribute("time")  = ts_values[population][2*(i-1)].c_str();
	      // conversion mV -> V
	      double V  = std::stod(ts_values[population][2*(i-1) + 1]);
	      V        -= population_V_shift_[population];
	      // 
	      time_series_node.append_attribute("V")     = V * 1.e-03;
	    }
	  

	  //
	  // clear the data
	  delete[] ts_data;
	  ts_data = nullptr;
	  // 
	  pch = nullptr;
	  //
	  population_rhythm_[population].clear();

	  // 
	  // Flush the population in the output file (memory regulation)
	  if( ( population % 100 == 0 && population != 0 ) || population == number_samples_ - 1 )
	    {
	      // 
	      std::string tempo_file = alpha_rhythm_output_file_;
	      tempo_file.insert( alpha_rhythm_output_file_.find(".xml"), 
				 std::string("_") + std::to_string(file_number++).c_str());
	      // 
	      document_.save_file( tempo_file.c_str() ); 
 	      // Reset for the next load
	      document_.reset();
	      fijee_ = document_.append_child("fijee");
	      dipoles_node = fijee_.append_child("dipoles");
	      dipoles_node.append_attribute("size") = static_cast<int>( populations_.size() );
 	    }
	} // end of for( int population = 0 ; population < number_samples_ ; population++ )


      //      // 
      //      // XML tail
      //      fwrite ( "</fijee>\n", sizeof(char), 9, out_XML_file_ );
      //
      //      // 
      //      // close the file
      //      fclose (out_XML_file_);
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
Biophysics::Brain_rhythm::output_electrodes_XML()
{
  //
  //
  std::cout << "XML output at the electrodes files" << std::endl;

  int file_number = 0;
  
  // 
  // Output XML file initialization
  auto setup_node = fijee_.append_child("setup");
  setup_node.append_attribute("size") = duration_ - 1;

  // 
  // 
  int index = 0;
  // 
  for( int i = 1 ; i < duration_ ; i++ )
    {
      auto electrodes_node = setup_node.append_child("electrodes");
      electrodes_node.append_attribute("index")  = index++;
      electrodes_node.append_attribute("dipole") = 0;
      electrodes_node.append_attribute("time")   = brain_rhythm_at_electrodes_[0][2*(i-1)];
      electrodes_node.append_attribute("size")   = number_electrodes_;
 
      // 
      for( int electrode = 0 ; electrode < number_electrodes_ ; electrode++ )
	{
	  // 
	  // 
	  auto electrode_node = electrodes_node.append_child("electrode");
	  // 
	  electrode_node.append_attribute("index")  = electrode;
	  electrode_node.append_attribute("label")  = leadfield_matrix_[electrode].get_label_().c_str();
	  electrode_node.append_attribute("V") = brain_rhythm_at_electrodes_[electrode][2*(i-1)+1];
	  electrode_node.append_attribute("I") = 0.;
	  // 
	}
      
      // 
      // Flush the output file (memory regulation)
      if( ( i % 100 == 0 && i != 0 ) || i == duration_ - 1 )
	{
	  // 
	  std::string tempo_file = alpha_rhythm_output_file_;
	  tempo_file.insert( alpha_rhythm_output_file_.find(".xml"), 
			     std::string("_electrodes_") + std::to_string(file_number++).c_str());
	  // 
	  document_.save_file( tempo_file.c_str() ); 
	  // Reset for the next load
	  document_.reset();
	  fijee_ = document_.append_child("fijee");
	  setup_node = fijee_.append_child("setup");
	  setup_node.append_attribute("size") = duration_ - 1;
	}
    }
}
