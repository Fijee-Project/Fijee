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
#include "PDE_solver_parameters.h"
//
// We give a comprehensive type name
//
typedef Solver::PDE_solver_parameters SDEsp;
typedef struct stat stat_file;
//
//
//
SDEsp*
SDEsp::parameters_instance_ = NULL;
//
//
//
SDEsp::PDE_solver_parameters()
{
  //
  // Check on ENV variables
  Fijee::Fijee_environment fijee;
  //
  files_path_         = fijee.get_fem_path_();
  files_path_output_  = fijee.get_fem_output_path_();
  files_path_result_  = fijee.get_fem_result_path_();
  files_path_measure_ = fijee.get_fem_measure_path_();
}
//
//
//
SDEsp::PDE_solver_parameters( const SDEsp& that ){}
//
//
//
SDEsp::~PDE_solver_parameters()
{}
//
//
//
SDEsp& 
SDEsp::operator = ( const SDEsp& that )
{
  //
  return *this;
}
//
//
//
SDEsp* 
SDEsp::get_instance()
{
  if( parameters_instance_ == NULL )
    parameters_instance_ = new SDEsp();
  //
  return parameters_instance_;
}
//
//
//
void 
SDEsp::kill_instance()
{
  if( parameters_instance_ != NULL )
    {
      delete parameters_instance_;
      parameters_instance_ = NULL;
    }
}
//
//
//
void 
SDEsp::init()
{
  try{
    // 
    // Read data set
    // 
    std::cout << "Load Fijee data set file" << std::endl;

    // 
    //
    pugi::xml_document     xml_file;
    pugi::xml_parse_result result = xml_file.load_file( "fijee.xml" );
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
	  // Get setup node
	  const pugi::xml_node setup_node = fijee_node.child("setup");
	  if (!setup_node)
	    {
	      std::cerr << "Read data from XML: no setup node" << std::endl;
	      exit(1);
	    }
	  // Get install directory
	  number_of_threads_ = setup_node.attribute("number_of_threads").as_int();
	  // 
	  if( number_of_threads_ < 1 )
	    {
	      std::string message = std::string("Error reading fijee.xml file.")
		+ std::string("The flag number_of_threads must be > 1.");
	      //
	      throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	    }

	  
	  //
	  // Get FEM node
	  const pugi::xml_node fem_node = fijee_node.child("fem");
	  if (!fem_node)
	    {
	      std::cerr << "Read data from XML: no fem node." << std::endl;
	      exit(1);
	    }

	  // 
	  // Get Solver node
	  const pugi::xml_node solver_node = fem_node.child("solver");
	  if (!solver_node)
	    {
	      std::cerr << "Read data from XML: no solver node." << std::endl;
	      exit(1);
	    }

	  // 
	  // Get linear algebra node
	  const pugi::xml_node linear_algebra_node = solver_node.child("la");
	  if (!linear_algebra_node)
	    {
	      std::cerr << "Read data from XML: no la node." << std::endl;
	      exit(1);
	    }
	  if(linear_algebra_node.attribute("iterative"))
	    {
	      // Back end parameters
	      dolfin::parameters["linear_algebra_backend"] = 
		linear_algebra_node.attribute("linear_algebra_backend").as_string();
	      // 
	      //
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
	      // cg - bicgstab - gmres
	      linear_solver_ = 
		std::string( linear_algebra_node.attribute("linear_solver").as_string() );
	      // ilut - ilu0 - block_ilu{t,0} - jacobi - row_scaling
	      preconditioner_ = 
		std::string( linear_algebra_node.attribute("preconditioner").as_string() );
	      //
	      maximum_iterations_ = linear_algebra_node.attribute("maximum_iterations").as_int();
	      // 
	      relative_tolerance_ = linear_algebra_node.attribute("relative_tolerance").as_double();
	    }
	  else
	    {
	      std::string message = std::string("Error reading fijee.xml file.")
		+ std::string(" Fijee is not yet prepared for Cholesky decomposition methods.");
	      //
	      throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	    }


	  // 
	  // Get Output node
	  const pugi::xml_node output_node = fem_node.child("output");
	  if (!output_node)
	    {
	      std::cerr << "Read data from XML: no output node." << std::endl;
	      exit(1);
	    }
	  // 
	  electric_current_density_field_ = 
	    output_node.attribute("electric_current_density_field").as_bool();
	  // 
	  electrical_field_ = 
	    output_node.attribute("electrical_field").as_bool();
	  // 
	  electric_potential_subdomains_ =
	    output_node.attribute("electric_potential_subdomains").as_bool();
	  // 
	  dipoles_electric_potential_ =
	    output_node.attribute("dipoles_electric_potential").as_bool();


	  //
	  //
	  break;
	};
      default:
	{
	  std::string message = std::string("Error reading fijee.xml file.")
	    + std::string(" You should look for an example in the 'share' directory located in the Fijee's install directory.");
	  //
	  throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	}
      }
  }
  catch( Fijee::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
}
//
//
//
std::ostream& 
Solver::operator << ( std::ostream& stream, 
		      const SDEsp& that)
{
  stream << " Pattern Singleton\n";
  //
  return stream;
}
