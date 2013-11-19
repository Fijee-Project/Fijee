#include <iostream>
#include "Subtraction.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::Subtraction::Subtraction()
{
  //
  // Load the mesh
  std::cout << "Load the mesh" << std::endl;
  mesh_.reset( new Mesh("../Mesh_rendering/mesh.xml") );
  //
  info( *mesh_ );

  //
  // Load Sub_domains
  std::cout << "Load Sub_domains" << std::endl;
  domains_.reset( new MeshFunction< long unsigned int >(*mesh_, "../Mesh_rendering/mesh_subdomains.xml") );

  //
  // Load the conductivity. Anisotrope conductivity
  std::cout << "Load the conductivity" << std::endl;
  sigma_.reset( new Solver::Tensor_conductivity(*mesh_) );




  //
  //
  //
  // Define the function space
  V_.reset( new Poisson::FunctionSpace(*mesh_) );
  
  //
  // Define boundary condition
  Periphery perifery;
  // Initialize mesh function for boundary domains. We tag the boundaries
  boundaries_.reset( new FacetFunction< size_t > (*mesh_) );
  boundaries_->set_all(0);
  perifery.mark(*boundaries_, 1);




  //
  // Read the dipoles xml file
  std::cout << "Load the dipoles" << std::endl;
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file("../Mesh_rendering/dipoles.xml");
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
	number_dipoles_ = dipoles_node.attribute("size").as_int();

	//
	//
	for( auto dipole : dipoles_node )
	  {
	    int index = dipole.attribute("index").as_uint();
	    // position
	    double position_x = dipole.attribute("x").as_double();
	    double position_y = dipole.attribute("y").as_double();
	    double position_z = dipole.attribute("z").as_double();
	    // Direction
	    double direction_vx = dipole.attribute("vx").as_double();
	    double direction_vy = dipole.attribute("vy").as_double();
	    double direction_vz = dipole.attribute("vz").as_double();
	    // Intensity
	    double Q = dipole.attribute("I").as_double();
	    // Conductivity
	    double lambda1 = dipole.attribute("lambda1").as_double();
	    double lambda2 = dipole.attribute("lambda2").as_double();
	    double lambda3 = dipole.attribute("lambda3").as_double();
	    //
	    dipoles_list_.push_back(std::move(Solver::Phi( index, Q,
							   position_x, position_y, position_z, 
							   direction_vx, direction_vy, direction_vz, 
							   lambda1, lambda2, lambda3 )));
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
  // check we read correctly the dipoles file
  if( number_dipoles_ != dipoles_list_.size() )
    {
      std::cerr << "The number of dipoles in the list is different from the number of dipoles in the file"
		<< std::endl;
      exit(1);
    }
}
//
//
//
void 
Solver::Subtraction::solver_loop()
{
//  //
//  // Define the function space
//  Poisson::FunctionSpace V(*mesh_);
//  
//  //
//  // Define boundary condition
//  Periphery perifery;
//  // Initialize mesh function for boundary domains. We tag the boundaries
//  FacetFunction< size_t > boundaries(*mesh_);
//  boundaries.set_all(0);
//  perifery.mark(boundaries, 1);

  //
  // Define the number of threads in the pool of threads
  Utils::Thread_dispatching pool( (SDEsp::get_instance())->get_number_of_threads_() );


  //
  //
  int tempo = 0;
  for( auto source = dipoles_list_.begin() ;
       source != dipoles_list_.end() ; source++ )
    if( ++tempo < 5)
    {
      //
      // Enqueue tasks
      pool.enqueue( std::ref(*this), std::ref(*source) );
    }
    else {break;}
}
//
//
//
void 
Solver::Subtraction::operator () ( Solver::Phi& source/*,
				   Poisson::FunctionSpace& V,
				   FacetFunction< size_t >& boundaries*/)
{
  //
  //
  std::cout << source.get_name_() << std::endl;
  
  //
  // Conductivity isotrope
  Solver::Sigma_isotrope a_inf( source.get_a0_() );

  //      //
  //      // Define Dirichlet boundary conditions 
  //      DirichletBC boundary_conditions(V, source, perifery);

  //////////////////////
  // Poisson equation //
  //////////////////////
      
  //
  // Define variational forms
  Poisson::BilinearForm a(*V_, *V_);
  Poisson::LinearForm L(*V_);
      
  //
  // Anisotropy
  // Bilinear
  a.a_sigma = *sigma_;
  a.dx      = *domains_;
  // Linear
  L.a_inf   =  a_inf;
  L.a_sigma = *sigma_;
  L.Phi_0   =  source;
  //
  L.dx    = *domains_;
  L.ds    = *boundaries_;

  //
  // Compute solution
  Function u(*V_);
  LinearVariationalProblem problem(a, L, u);
  LinearVariationalSolver  solver(problem);
  // krylov
  solver.parameters["linear_solver"]  
    = (SDEsp::get_instance())->get_linear_solver_();
  solver.parameters("krylov_solver")["maximum_iterations"] 
    = (SDEsp::get_instance())->get_maximum_iterations_();
  solver.parameters("krylov_solver")["relative_tolerance"] 
    = (SDEsp::get_instance())->get_relative_tolerance_();
  solver.parameters["preconditioner"] 
    = (SDEsp::get_instance())->get_preconditioner_();
  //
  solver.solve();

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
  std::string file_name = source.get_name_() + std::string(".pvd");
  File file( file_name.c_str() );
  //
  file << u;
  file << *domains_;
};
