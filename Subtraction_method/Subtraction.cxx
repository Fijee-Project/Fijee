#include <iostream>
#include"Subtraction.h"


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
  //
  // Define the function space
  Poisson::FunctionSpace V(*mesh_);
  
  //
  // Define boundary condition
  Periphery perifery;
  // Initialize mesh function for boundary domains. We tag the boundaries
  FacetFunction< size_t > boundaries(*mesh_);
  boundaries.set_all(0);
  perifery.mark(boundaries, 1);
  
  //
  //
  bool tempo = true;
  for( auto source : dipoles_list_ )
    if(tempo)
    {
      tempo = false;
      //
      //
      std::cout << "Dipole: " << source.get_index_() << std::endl;
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
      Poisson::BilinearForm a(V, V);
      Poisson::LinearForm L(V);
      
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
      L.ds    =  boundaries;

      //
      // Compute solution
      u_.reset( new Function(V) );
      LinearVariationalProblem problem(a, L, *u_);
      LinearVariationalSolver  solver(problem);
      // krylov

      //    krylov_solver            |    type  value          range  access  change
      //    ------------------------------------------------------------------------
      //    absolute_tolerance       |  double  1e-15             []       0       0
      //    divergence_limit         |  double  10000             []       0       0
      //    error_on_nonconvergence  |    bool   true  {true, false}       0       0
      //    maximum_iterations       |     int  10000             []       0       0
      //    monitor_convergence      |    bool  false  {true, false}       0       0
      //    nonzero_initial_guess    |    bool  false  {true, false}       0       0
      //    relative_tolerance       |  double  1e-06             []       0       0
      //    report                   |    bool   true  {true, false}       0       0
      //    use_petsc_cusp_hack      |    bool  false  {true, false}       0       0
      //
      solver.parameters["linear_solver"]  = "cg";
      solver.parameters("krylov_solver")["maximum_iterations"] = 20000;
      //  solver.parameters["linear_solver"]  = "bicgstab";
      //  solver.parameters["linear_solver"]  = "cg";
      solver.parameters["preconditioner"] = "ilu";
      // Cholesky
      //  solver.parameters["linear_solver"]  = "umfpack";
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

      File 
	file_pvd("poisson.pvd"),
	file_xyz("poisson.xyz"),
	file_xml("poisson.xml"),
	file_svg("poisson.svg"),
	file_raw("poisson.raw"),
	file_bin("poisson.bin");
      //
      file_pvd << *u_;
      file_xyz << *u_;
      file_xml << *u_;
      file_svg << *u_;
      file_raw << *u_;
      file_bin << *u_;
      //      file << *domains_;

      //
      // rekease the solution for the next calculation
      u_.reset();
    }
}
