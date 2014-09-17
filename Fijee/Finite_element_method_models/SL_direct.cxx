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
#include "SL_direct.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::SL_direct::SL_direct():Physics()
{
  //
  // Define the function space
  V_.reset( new SLD_model::FunctionSpace(*mesh_) );
  
  //
  // Define boundary condition
  perifery_.reset( new Periphery() );
  // Initialize mesh function for boundary domains. We tag the boundaries
  boundaries_.reset( new FacetFunction< size_t > (*mesh_) );
  boundaries_->set_all(0);
  perifery_->mark(*boundaries_, 1);


  //
  // Read the dipoles xml file
  std::cout << "Load dipoles file" << std::endl;
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
	    // Index cell
	    double index_cell = dipole.attribute("index_cell").as_uint();
	    //
	    dipoles_list_.push_back(std::move(Solver::Current_density( index, index_cell, Q,
								       position_x, position_y, position_z, 
								       direction_vx, direction_vy, direction_vz )));
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
Solver::SL_direct::operator () ( /*Solver::Phi& source,
				   SLD_model::FunctionSpace& V,
				   FacetFunction< size_t >& boundaries*/)
{
  //
  // Mutex the dipoles vector poping process
  //
  Solver::Current_density source;
    try 
      {
	// lock the dipole list
	std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	source = dipoles_list_.front();
	dipoles_list_.pop_front();
      }
    catch (std::logic_error&) 
      {
	std::cerr << "[exception caught]\n" << std::endl;
      }

  //
  //
  std::cout << source.get_name_() << std::endl;
  
//  //
//  // Define Dirichlet boundary conditions 
//  DirichletBC boundary_conditions(*V, source, perifery);


  ///////////////////////////////////////////////
  // Source localization direct model equation //
  ///////////////////////////////////////////////
      
  //
  // Define variational forms
  SLD_model::BilinearForm a(*V_, *V_);
  SLD_model::LinearForm L(*V_);
      
  //
  // Anisotropy
  // Bilinear
  a.a_sigma  = *sigma_;
  a.dx       = *domains_;
  // Linear
  L.J_source = source;
  //
  L.dx       = *domains_;
  L.ds       = *boundaries_;

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
  // Save solution in VTK format
  //  * Binary (.bin)
  //  * RAW    (.raw)
  //  * SVG    (.svg)
  //  * XD3    (.xd3)
  //  * XDMF   (.xdmf)
  //  * XML    (.xml)
  //  * XYZ    (.xyz)
  //  * VTK    (.pvd)
  std::string file_name = (SDEsp::get_instance())->get_files_path_result_() + source.get_name_() + std::string(".pvd");
  File file( file_name.c_str() );
  //
  file << u;
};
