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
  Utils::Fijee_environment fijee;
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
  //
  //  dolfin::parameters["num_threads"] = 4;

  //
  // Back end parameters
  // Allowed values are: [PETSc, STL, uBLAS, Epetra, MTL4, ViennaCL].
  // Epetra in Trilinos
  // uBLAS needs UMFPACK
  dolfin::parameters["linear_algebra_backend"] = "ViennaCL";
  //  info(solver.parameters,true) ;
  //  info(parameters,true) ;

  //
  // Cholesky: umfpack


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
  linear_solver_ = "cg";
  // ilut - ilu0 - block_ilu{t,0} - jacobi - row_scaling
  preconditioner_ = "row_scaling";
  //
  maximum_iterations_ = 10000000;
  //
  //  relative_tolerance_ = 1.e-8 /*1.e-8*/;
//  relative_tolerance_ = 7.e-4 /*tDCS_spheres*/;
//  relative_tolerance_ = 1.e-8 /*tDCS_spheres*/;
//  relative_tolerance_ = 5.e-3 /*tDCS_head*/;
  relative_tolerance_ = 1.e-2 /*tDCS_head*/;

  //
  // Dispatching information
  number_of_threads_ = 2;
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
