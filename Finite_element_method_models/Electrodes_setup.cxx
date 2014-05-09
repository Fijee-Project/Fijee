#include "Electrodes_setup.h"

typedef Solver::PDE_solver_parameters SDEsp;

/* Electrodes_setup */
//
//
//
Solver::Electrodes_setup::Electrodes_setup()
{
  //
  // Read the electrodes xml file
  std::cout << "Load electrodes file" << std::endl;
  //
  std::string electrodes_xml = (SDEsp::get_instance())->get_files_path_output_();
  electrodes_xml += "electrodes.xml";
  //
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file( electrodes_xml.c_str() );
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
	number_samples_ = setup_node.attribute("size").as_int();
	current_setup_.resize( number_samples_ );
	// loop over the samples
	for ( auto sample : setup_node )
	  {
	    //
	    // Get the number of electrodes
	    int sample_number  = sample.attribute("index").as_int();
	    double sample_time = sample.attribute("time").as_double();
	    number_electrodes_ = sample.attribute("size").as_int();
	    current_setup_[sample_number].reset( new Solver::Electrodes_injection(sample_time) );

	    //
	    //
	    for( auto electrode : sample )
	      {
		int index = electrode.attribute("index").as_uint();
		// position
		double position_x = electrode.attribute("x").as_double(); /* m */
		double position_y = electrode.attribute("y").as_double(); /* m */
		double position_z = electrode.attribute("z").as_double(); /* m */
		// Direction
		double direction_vx = electrode.attribute("vx").as_double();
		double direction_vy = electrode.attribute("vy").as_double();
		double direction_vz = electrode.attribute("vz").as_double();
		// Label
		std::string label = electrode.attribute("label").as_string(); 
		// Intensity
		double I = electrode.attribute("I").as_double(); /* Ampere */
		// Potential
		double V = electrode.attribute("V").as_double(); /* Volt */
		// Impedance
		double Re_z_l = electrode.attribute("Re_z_l").as_double();
		double Im_z_l = electrode.attribute("Im_z_l").as_double();
		// Contact surface between Electrode and the scalp
		double surface = electrode.attribute("surface").as_double(); /* m^2 */
		double radius  = electrode.attribute("radius").as_double();  /* m */
		//
		current_setup_[sample_number]
		  ->add_electrode( "Current", index/*, index_cell*/, label, I,
				   Point(position_x, position_y, position_z), 
				   Point(direction_vx, direction_vy, direction_vz),
				   Re_z_l, Im_z_l, surface, radius );
	      }
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
//
//
//
bool
Solver::Electrodes_setup::inside( const Point& Vertex ) const
{
  return current_setup_[0]->inside(Vertex);
}
//
//
//
bool
Solver::Electrodes_setup::add_potential_value( const std::string Electrode_label, const double U ) 
{
  return current_setup_[0]->add_potential_value(Electrode_label, U);
}
//
//
//
bool
Solver::Electrodes_setup::add_potential_value( const Point& Vertex, const double U ) 
{
  return current_setup_[0]->add_potential_value(Vertex, U);
}
//
//
//
std::tuple<std::string, bool> 
Solver::Electrodes_setup::inside_probe( const Point& Vertex ) const
{
  return current_setup_[0]->inside_probe(Vertex);
}
//
//
//
void
Solver::Electrodes_setup::set_boundary_cells( const std::map<std::string, 
					      std::map<std::size_t, 
					      std::list< MeshEntity  >  >  >& Map_electrode_cells  )
{
  for ( auto sample = current_setup_.begin() ; sample != current_setup_.end() ; sample++ )
    (*sample)->set_boundary_cells( Map_electrode_cells );
}
