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
#ifndef PARCELLATION_INFORMATION_H
#define PARCELLATION_INFORMATION_H
#include <list>
//
// FEniCS
//
#include <dolfin.h>
//
//
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "PDE_solver_parameters.h"
#include "Utils/XML_writer.h"
//
//
//
typedef Solver::PDE_solver_parameters SDEsp;
//
//
//
/*!
 * \file Parcellation_information.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Solver
{
  /*! \class Parcellation_information
   * \brief class representing information at the centroid of a parcel
   *
   *  This class is an example of class I will have to use
   */
  template < typename Physical_value >
    class Parcellation_information : public Utils::XML_writer
    {
    private:
    //! List of parcels
    std::list< Physical_value> parcels_list_;
      
    public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Parcellation_information
     *
     */
    Parcellation_information( const std::string& );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Parcellation_information( const Parcellation_information& ){};
    /*!
     *  \brief Destructor
     *
     *  Destructor of the class Parcellation_information
     *
     */
    ~Parcellation_information(){/*Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Parcellation_information
     *
     */
    Parcellation_information& operator = ( const Parcellation_information& ){return *this;};
    };
  // 
  // 
  // 
  template < typename Physical_value >
    Parcellation_information< Physical_value >::Parcellation_information( const std::string& Phys_value ):
    Utils::XML_writer("")
  {
    // 
    // Output file
    std::string parcellation_output = (SDEsp::get_instance())->get_files_path_result_();
    parcellation_output += Phys_value + std::string("_parcellation.xml");
    // 
    set_file_name_( parcellation_output );

    //
    // Read the parcellation xml file
    std::cout << "Load parcellation file" << std::endl;
    //
    std::string dipoles_xml = (SDEsp::get_instance())->get_files_path_output_();
    dipoles_xml += "parcellation.xml";
    //
    pugi::xml_document     xml_file;
    pugi::xml_parse_result result = xml_file.load_file( dipoles_xml.c_str() );
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
	  // Get dipoles node
	  const pugi::xml_node dipoles_node = fijee_node.child("dipoles");
	  if (!dipoles_node)
	    {
	      std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	      exit(1);
	    }
	  // Get the number of dipoles
//	  number_dipoles_ = dipoles_node.attribute("size").as_int();
	  
	  //
	  //
	  for( auto dipole : dipoles_node )
	    parcels_list_.push_back(std::move(Physical_value( dipole.attribute("index").as_uint(), // index corresponding to the parcel
							      dipole.attribute("x").as_double(),   // position
							      dipole.attribute("y").as_double(),   // position
							      dipole.attribute("z").as_double(),   // position
							      dipole.attribute("vx").as_double(),  // direction
							      dipole.attribute("vy").as_double(),  // direction
							      dipole.attribute("vz").as_double(),  // direction
							      dipole.attribute("I").as_double(),   // current
							      dipole.attribute("index_cell").as_uint(),   // index cell
							      dipole.attribute("index_parcel").as_uint(), // index parcel
							      dipole.attribute("lambda1").as_double(), // Lambda
							      dipole.attribute("lambda2").as_double(), // Lambda
							      dipole.attribute("lambda3").as_double()  // Lambda
							      )));
	  
	  //
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
}
#endif
