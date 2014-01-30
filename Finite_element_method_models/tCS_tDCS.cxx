#include <iostream>
#include "tCS_tDCS.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::tCS_tDCS::tCS_tDCS()
{
  //
  // Load the mesh
  std::cout << "Load the mesh" << std::endl;
  //
  std::string mesh_xml = (SDEsp::get_instance())->get_files_path_output_();
  mesh_xml += "mesh.xml";
  //
  mesh_.reset( new Mesh(mesh_xml.c_str()) );
  //
  info( *mesh_ );

  //
  // Load Sub_domains
  std::cout << "Load Sub_domains" << std::endl;
  //
  std::string subdomains_xml = (SDEsp::get_instance())->get_files_path_output_();
  subdomains_xml += "mesh_subdomains.xml";
  //
  domains_.reset( new MeshFunction< long unsigned int >(*mesh_, subdomains_xml.c_str()) );
  // write domains
  std::string domains_file_name = (SDEsp::get_instance())->get_files_path_result_();
  domains_file_name            += std::string("domains.pvd");
  File domains_file( domains_file_name.c_str() );
  domains_file << *domains_;


  //
  // Load the conductivity. Anisotrope conductivity
  std::cout << "Load the conductivity" << std::endl;
  sigma_.reset( new Solver::Tensor_conductivity(*mesh_) );


  //
  //
  //
  // Define the function space
  V_.reset( new tCS_model::FunctionSpace(*mesh_) );
  
  //
  // Define boundary condition
  perifery_.reset( new Periphery() );
  // Initialize mesh function for boundary domains. We tag the boundaries
  boundaries_.reset( new FacetFunction< size_t > (*mesh_) );
  boundaries_->set_all(0);
  perifery_->mark(*boundaries_, 1);


  //
  // Read the electrodes xml file
  std::cout << "Load the electrodes" << std::endl;
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
	// Get electrodes node
	const pugi::xml_node electrodes_node = fijee_node.child("electrodes");
	if (!electrodes_node)
	  {
	    std::cerr << "Read data from XML: no electrodes node" << std::endl;
	    exit(1);
	  }
	// Get the number of electrodes
	number_electrodes_ = electrodes_node.attribute("size").as_int();

	//
	//
	for( auto electrode : electrodes_node )
	  {
	    int index = electrode.attribute("index").as_uint();
	    // position
	    double position_x = electrode.attribute("x").as_double(); /* mm */
	    double position_y = electrode.attribute("y").as_double(); /* mm */
	    double position_z = electrode.attribute("z").as_double(); /* mm */
	    // Direction
	    double direction_vx = electrode.attribute("vx").as_double();
	    double direction_vy = electrode.attribute("vy").as_double();
	    double direction_vz = electrode.attribute("vz").as_double();
	    // Label
	    std::string label = electrode.attribute("label").as_string(); 
	    // Intensity
	    double I = electrode.attribute("I").as_double(); /* Ampere */
	    // Impedance
	    double Re_z_l = electrode.attribute("Re_z_l").as_double();
	    double Im_z_l = electrode.attribute("Im_z_l").as_double();
	    // Contact surface between Electrode and the scalp
	    double surface = electrode.attribute("surface").as_double(); /* m^2 */
//	    // Index cell
//	    double index_cell = electrode.attribute("index_cell").as_uint();
	    //
	    electrodes_vector_.push_back(std::move(Solver::Current_intensity( index/*, index_cell*/, label, I, 
									      position_x, position_y, position_z, 
									      direction_vx, direction_vy, direction_vz,
									      Re_z_l, Im_z_l, surface )));
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

  //
  // check we read correctly the electrodes file
  if( number_electrodes_ != electrodes_vector_.size() )
    {
      std::cerr << "The number of electrodes in the list is different from the number of electrodes in the file"
		<< std::endl;
      exit(1);
    }
}
//
//
//
void 
Solver::tCS_tDCS::operator () ( /*Solver::Phi& source,
				        SLD_model::FunctionSpace& V,
				        FacetFunction< size_t >& boundaries*/)
{
//  //
//  // Mutex the electrodes vector poping process
//  //
//  Solver::Current_density source;
//    try {
//      // lock the electrode list
//      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
//      source = electrodes_list_.front();
//      electrodes_list_.pop_front();
//    }
//    catch (std::logic_error&) {
//      std::cout << "[exception caught]\n";
//    }
//
//  //
//  //
//  std::cout << source.get_name_() << std::endl;
//  
////  //
////  // Define Dirichlet boundary conditions 
////  DirichletBC boundary_conditions(*V, source, perifery);


  //////////////////////////////////////////////////////
  // Transcranial direct current stimulation equation //
  //////////////////////////////////////////////////////
      
  //
  // Define variational forms
  tCS_model::BilinearForm a(*V_, *V_);
  tCS_model::LinearForm L(*V_);
      
 //
 // Anisotropy
 // Bilinear
 a.a_sigma  = *sigma_;
 a.dx       = *domains_;
 // Linear
 // Vector I(electrodes_vector_);
// L.I = I;
////  Constant Cte (0.);
////  L.Cte      = Cte;
////  L.ds       = *boundaries_;
////
//  //
//  // Compute solution
//  Function u(*V_);
//  LinearVariationalProblem problem(a, L, u);
//  LinearVariationalSolver  solver(problem);
//  // krylov
//  solver.parameters["linear_solver"]  
//    = (SDEsp::get_instance())->get_linear_solver_();
//  solver.parameters("krylov_solver")["maximum_iterations"] 
//    = (SDEsp::get_instance())->get_maximum_iterations_();
//  solver.parameters("krylov_solver")["relative_tolerance"] 
//    = (SDEsp::get_instance())->get_relative_tolerance_();
//  solver.parameters["preconditioner"] 
//    = (SDEsp::get_instance())->get_preconditioner_();
//  //
//  solver.solve();

  //
  // Save solution in VTK format
  //  * Binary (.bin)
  //  * RAW    (.raw)
  //  * SVG    (.svg)
  //  * XD3    (.xd3)
  //  * XDMF   (.xdmf)
  //  * XML    (.xml)
  //  * XYZ    (.xyz)
  //  * VTK    (.pvd)
//  std::string file_name = (SDEsp::get_instance())->get_files_path_result_() + 
//  source.get_name_() + std::string(".pvd");
//  File file( file_name.c_str() );
//  //
//  file << u;
};
