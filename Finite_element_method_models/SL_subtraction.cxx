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
#include <iostream>
#include "SL_subtraction.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::SL_subtraction::SL_subtraction():Physics()
{
  //
  // Define the function space
  V_.reset( new SLS_model::FunctionSpace(mesh_) );
  // 
  a_.reset( new SLS_model::BilinearForm(V_, V_) );
  // 
  initialized_ = false;
  
  //
  // Define boundary condition
  Periphery perifery;
  // Initialize mesh function for boundary domains. We tag the boundaries
  boundaries_.reset( new FacetFunction< size_t > (*mesh_) );
  boundaries_->set_all(0);
  perifery.mark(*boundaries_, 1);


  //
  // Read the dipoles xml file
  std::cout << "Load dipoles file" << std::endl;
  //
  std::string dipoles_xml = (SDEsp::get_instance())->get_files_path_output_();
  //  dipoles_xml += "dipoles.xml";
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
  
  // 
  // Reset outputfile
  std::string leadfield_matrix_xml = (SDEsp::get_instance())->get_files_path_result_();
  leadfield_matrix_xml += "leadfield_matrix.xml";

  electrodes_->set_file_name_(leadfield_matrix_xml);
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
    try 
      {
	// lock the dipole list
	std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	source = dipoles_list_.front();
	dipoles_list_.pop_front();
	// 
	if( !initialized_ )
	  {
	    //
	    // Anisotropy conductivity
	    // Bilinear form
	    a_->a_sigma = *sigma_;
	    //  a.dx      = *domains_;
	    //
	    A_.reset( new Matrix() );
	    assemble(*A_, *a_);

	    // 
	    // 
	    initialized_ = true;
	  }
      }
    catch (std::logic_error&) 
      {
	std::cerr << "[exception caught]\n" << std::endl;
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
  //  SLS_model::BilinearForm a(V_, V_);
  SLS_model::LinearForm L(V_);
  Vector L_;

  // Linear
  L.a_inf   =  a_inf;
  L.a_sigma = *sigma_;
  L.Phi_0   =  source;
  //
  //  L.dx    = *domains_;
  //  L.ds    = *boundaries_;
  // Assembling the linear form in a vector
  assemble(L_, L);

  //
  // Compute solution
  Function u(V_);
  //
// ORIG  LinearVariationalProblem problem(*a_, L, u);
// ORIG  LinearVariationalSolver  solver(problem);
// ORIG  // krylov
// ORIG  solver.parameters["linear_solver"]  
// ORIG    = (SDEsp::get_instance())->get_linear_solver_();
// ORIG  solver.parameters("krylov_solver")["maximum_iterations"] 
// ORIG    = (SDEsp::get_instance())->get_maximum_iterations_();
// ORIG  solver.parameters("krylov_solver")["relative_tolerance"] 
// ORIG    = (SDEsp::get_instance())->get_relative_tolerance_();
// ORIG  solver.parameters["preconditioner"] 
// ORIG    = (SDEsp::get_instance())->get_preconditioner_();
// ORIG  //
// ORIG  solver.solve();

  
  // 
  //
  KrylovSolver solver( (SDEsp::get_instance())->get_linear_solver_(),
		       (SDEsp::get_instance())->get_preconditioner_() );
  // Set parameters of the Krylov solver
  solver.parameters["maximum_iterations"] 
    = (SDEsp::get_instance())->get_maximum_iterations_();
  solver.parameters["relative_tolerance"] 
    = (SDEsp::get_instance())->get_relative_tolerance_();
  // 
  solver.solve( *A_, *u.vector(), L_ );


  //
  // Regulation terme:  \int u dx = 0
  double old_u_bar = 0.;
  double u_bar = 1.e+6;
  double U_bar = 0.;
  double N = u.vector()->size();
  int iteration = 0;
  double Sum = 1.e+6;
  //
  while ( fabs(Sum) > 1.e-6 )
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
  if ( false )
    {
      std::list<std::size_t> test_sub_domains{4,5};
      solution_domain_extraction(Phi_tot, test_sub_domains, source.get_name_().c_str());
      
    }

  //
  // Mutex record potential at each electrods
  try 
    {
      // lock the dipole list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      // 
      //      electrodes_->get_current(0)->punctual_potential_evaluation(Phi_tot/*u*/, mesh_);
      electrodes_->get_current(0)->surface_potential_evaluation(Phi_tot/*u*/, mesh_);
      electrodes_->record_potential( /*dipole idx*/ source.get_index_(), 
				     /*time   idx*/ 0);
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }

  if( false )
    {
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
    }
};
