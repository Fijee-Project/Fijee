#include <iostream>
#include "SL_subtraction.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::SL_subtraction::SL_subtraction():Physics()
{
  //
  //
  //
  // Define the function space
  V_.reset( new SLS_model::FunctionSpace(mesh_) );
  
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
  //
  std::string dipoles_xml = (SDEsp::get_instance())->get_files_path_output_();
  dipoles_xml += "dipoles.xml";
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
	    dipoles_list_.push_back(/*std::move(*/Solver::Phi( index, Q,
							       position_x, position_y, position_z, 
							       direction_vx, direction_vy, direction_vz, 
							       lambda1, lambda2, lambda3 )/*)*/);
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
Solver::SL_subtraction::operator () ( /*Solver::Phi& source,
				        SLS_model::FunctionSpace& V,
				        FacetFunction< size_t >& boundaries*/)
{
  //
  // Mutex the dipoles vector poping process
  //
  Solver::Phi source;
    try {
      // lock the dipole list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      source = dipoles_list_.front();
      dipoles_list_.pop_front();
    }
    catch (std::logic_error&) {
      std::cout << "[exception caught]\n";
    }

  //
  //
  std::cout << source.get_name_() << std::endl;
  
  //
  // Conductivity isotrope
  Solver::Sigma_isotrope a_inf( source.get_a0_() );

  //      //
  //      // Define Dirichlet boundary conditions 
  //      DirichletBC boundary_conditions(*V, source, perifery);

  ////////////////////////////////////////////////////
  // Source localization subtraction model equation //
  ////////////////////////////////////////////////////
      
  //
  // Define variational forms
  SLS_model::BilinearForm a(V_, V_);
  SLS_model::LinearForm L(V_);

  //
  // Anisotropy
  // Bilinear
  a.a_sigma = *sigma_;
  //  a.dx      = *domains_;
  // Linear
  L.a_inf   =  a_inf;
  L.a_sigma = *sigma_;
  L.Phi_0   =  source;
  //
  //  L.dx    = *domains_;
  //  L.ds    = *boundaries_;

  //
  // Compute solution
  Function u(V_);
  //
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
 // Regulation terme:  \int u dx = 0
 double old_u_bar = 0.;
 double u_bar = 1.e+6;
 double U_bar = 0.;
 double N = u.vector()->size();
 int iteration = 0;
 double Sum = 1.e+6;
 //
 //  while ( abs( u_bar - old_u_bar ) > 0.1 )
 while ( abs(Sum) > 1.e-3 )
   {
     old_u_bar = u_bar;
     u_bar  = u.vector()->sum();
     u_bar /= N;
     (*u.vector()) -= u_bar;
     //
     U_bar += u_bar;
     Sum = u.vector()->sum();
     std::cout << ++iteration << " ~ " << Sum  << std::endl;
   }
 
 std::cout << "int u dx = " << Sum << std::endl;

  //
  // V_{tot} = \sum_{i=1}^{n} U_{i} \phi_{i}. where \{\phi_i\}_{i=1}^{n} is a basis for V_h, 
  // and U is a vector of expansion coefficients for V_{tot,h}.
  Function Phi_tot(V_);
  Phi_tot.interpolate(source);
  *Phi_tot.vector()  += *u.vector();


 //
 // Filter function over a subdomain
 std::list<std::size_t> test_sub_domains{4,5};
 solution_domain_extraction(Phi_tot, test_sub_domains, "Source_localization");
  


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
  std::string file_name = (SDEsp::get_instance())->get_files_path_result_() 
  + source.get_name_() + std::string(".pvd");
  File file( file_name.c_str() );
//  std::string file_th_name = source.get_name_() + std::string("_Phi_th.pvd");
//  File file_th(file_th_name.c_str());
//  std::string file_tot_name = source.get_name_() + std::string("_Phi_tot.pvd");
//  File file_tot(file_tot_name.c_str());
  //
//  file << u;
//  file_th << Phi_th;
  file << Phi_tot;
};
