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
Utils::Biophysics::Brain_rhythm::Brain_rhythm():
  Utils::XML_writer("alpha_rhythm.xml")
{}
// 
// 
// 
void
Utils::Biophysics::Brain_rhythm::load_electrode_file( std::string In_electrode_file_XML )
{
  //
  // Read the electrodes xml file
  std::cout << "Load electrodes file for alpha generation" << std::endl;
  //
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file( In_electrode_file_XML.c_str() );
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
	const pugi::xml_node setup_node = fijee_node.child("setup");
	if (!setup_node)
	  {
	    std::cerr << "Read data from XML: no setup node" << std::endl;
	    exit(1);
	  }
	// Get the number of samples
	//	number_samples_ = setup_node.attribute("size").as_int();
	// loop over the samples
	for ( auto sample : setup_node )
	  {
	    //
	    // Get the number of electrodes
	    // int sample_number = sample.attribute("index").as_int();
	    int number_electrodes = sample.attribute("size").as_int();

	    //
	    //
	    electrode_rhythm_.resize(number_electrodes);
	    electrode_mapping_.resize(number_electrodes);
	    // 
	    for( auto electrode : sample )
	      {
		int index = electrode.attribute("index").as_uint();
		// Label
		std::string label = electrode.attribute("label").as_string(); 
		// 
		electrode_mapping_[index] = label;
	      }
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

  // 
  // Output XML file initialization
  // 
  setup_node_ = fijee_.append_child("setup");
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
      // get the time
      output_stream_ <<  std::get<0>( *(it[0]) ) << " ";
      // 
     for (int electrode = 0 ; electrode < number_of_electrodes ; electrode++)
       output_stream_ <<  std::get<1>( *(it[electrode]++) ) << " ";
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
Utils::Biophysics::Brain_rhythm::output_XML()
{
  // 
  // Build XML output 
  // 
  
  // Size of the time series
  setup_node_.append_attribute("size") = static_cast<int>( electrode_rhythm_[0].size() );
  // 
  int number_of_electrodes = electrode_rhythm_.size();

  // 
  // loop over the time series
  std::vector< std::list< std::tuple< double, double, double > >::const_iterator > 
    it(number_of_electrodes);
  // inializes all the lists
  for( int electrode = 0 ; electrode < number_of_electrodes ; electrode++ )
    it[electrode] = electrode_rhythm_[electrode].begin();
  // 
  int index = 0;
  while( it[0] !=  electrode_rhythm_[0].end())
    {
      auto electrodes_node = setup_node_.append_child("electrodes");
      // 
      electrodes_node.append_attribute("index")  = index++;
      electrodes_node.append_attribute("dipole") = 0;
      electrodes_node.append_attribute("time")   = std::get<0>( *(it[0]) );
      electrodes_node.append_attribute("size")   = number_of_electrodes;

      //
      // loop over electrodes  
      for ( int electrode = 0 ; electrode < number_of_electrodes ; electrode++ )
	{
	  auto electrode_node = electrodes_node.append_child("electrode");
	  // 
	  electrode_node.append_attribute("index") = electrode;
	  electrode_node.append_attribute("label") = electrode_mapping_[electrode].c_str();
	  electrode_node.append_attribute("V")     = std::get<1>( *(it[electrode]++) );
	  electrode_node.append_attribute("I")     = 0.0;
	}
    }
  

  // 
  // Statistical analysise
  Make_analysis();
}
