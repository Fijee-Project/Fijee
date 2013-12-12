#include "Build_electrodes_list.h"
//
// UCSF
//
#include "Utils/enum.h"
#include "Labeled_domain.h"
#include "VTK_implicite_domain.h"
//
//
//
#define PI 3.14159265359
//
// We give a comprehensive type name
//
typedef Domains::Build_electrodes_list DBel;
//
//
//
DBel::Build_electrodes_list()
{
  //
  // Load the MNI electrode positions
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file( (Domains::Access_parameters::get_instance())->get_electrodes_10_20_() );

  int number_electrods_;
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
	const pugi::xml_node electrodes_node = fijee_node.child("electrodes");
	if (!electrodes_node)
	  {
	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	    exit(1);
	  }
	// Get the number of dipoles
	number_electrods_ = electrodes_node.attribute("size").as_int();

	//
	//
	for( auto electrode : electrodes_node )
	  {
	    int index = electrode.attribute("index").as_uint();
	    // Name of the electrode
	    std::string label = electrode.attribute("label").as_string();
	    // Cartesian position
	    double position_x = electrode.attribute("x").as_double();
	    double position_y = electrode.attribute("y").as_double();
	    double position_z = electrode.attribute("z").as_double();
	    // Spherical position
	    double phi    = electrode.attribute("phi").as_double();
	    double theta  = electrode.attribute("theta").as_double();
	    double radius = electrode.attribute("radius").as_double();
	    // move in a regular spherical frame work
	    // x = r * cos(phi) * sin(theta)
	    // y = r * sin(phi) * sin(theta)
	    // z = r * cos(theta)
	    phi = phi * PI / 180.;
	    theta = PI/2. - theta * PI / 180.;
	    //
	    electrodes_.push_back( Domains::Electrode( index, label, 
						       position_x, position_y, position_z,
						       radius, phi, theta ) );
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
////
////
////
//DBel::Build_electrodes_list( const DBel& that )
//{
//}
////
////
////
//DBel& 
//DBel::operator = ( const DBel& that )
//{
//
//  //
//  //
//  return *this;
//}
//
//
//
void
DBel::adjust_cap_positions_on( Labeled_domain< VTK_implicite_domain, GT::Point_3, std::list< Point_vector > >&  Sclap )
{
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DBel& that)
{
//  std::for_each( that.get_list_position().begin(),
//		 that.get_list_position().end(),
//		 [&stream]( int Val )
//		 {
//		   stream << "list pos = " << Val << "\n";
//		 });
//  //
//  stream << "position x = " <<    that.get_pos_x() << "\n";
//  stream << "position y = " <<    that.get_pos_y() << "\n";
//  if ( &that.get_tab() )
//    {
//      stream << "tab[0] = "     << ( &that.get_tab() )[0] << "\n";
//      stream << "tab[1] = "     << ( &that.get_tab() )[1] << "\n";
//      stream << "tab[2] = "     << ( &that.get_tab() )[2] << "\n";
//      stream << "tab[3] = "     << ( &that.get_tab() )[3] << "\n";
//    }
  //
  return stream;
};
