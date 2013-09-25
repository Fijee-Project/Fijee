#include <iostream>
#include"Sources.h"


//
//
//
Solver::Sources::Sources()
{
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file("Dipoles_distribution.xml");
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
	// Get distribution node
	const pugi::xml_node distribution_node = fijee_node.child("distribution");
	if (!distribution_node)
	  {
	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	    exit(1);
	  }
  
	//
	// Get dipoles node
	const pugi::xml_node dipoles_node = distribution_node.child("dipoles");
	if (!dipoles_node)
	  {
	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	    exit(1);
	  }
	// Get the number of dipoles
	number_dipoles_ = dipoles_node.attribute("size").as_int();

	//
	// Iterate over dipoles and add to 
//	if( number_dipoles_ != dipoles_node.size() )
//	  {
//	    std::cerr << "Read data from XML: dipoles size mismatch" << std::endl;
//	    exit(1);
//	  }
	//
	for( auto dipole : dipoles_node )
	  {
	    dipole.attribute("index").as_uint();
	    // position
	    dipole.attribute("x").as_double();
	    dipole.attribute("Y").as_double();
	    dipole.attribute("Z").as_double();
	    // DIRECTION
	    dipole.attribute("nx").as_double();
	    dipole.attribute("ny").as_double();
	    dipole.attribute("nz").as_double();
	  }
	
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
