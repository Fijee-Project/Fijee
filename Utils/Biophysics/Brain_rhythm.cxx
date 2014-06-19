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
  setup_node_.append_attribute("size") = static_cast<int>(electrode_mapping_.size());
}
