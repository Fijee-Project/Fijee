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
#include <vector>
#include <memory>
//
// UCSF
//
#include "PDE_solver_parameters.h"
#include "SL_subtraction.h"
#include "SL_direct.h"
#include "tCS_tDCS.h"
#include "tCS_tACS.h"
#include "tCS_tDCS_local_conductivity.h"
#include "Model_solver.h"

int main()
{
  //
  //
  Solver::PDE_solver_parameters* solver_parameters = Solver::PDE_solver_parameters::get_instance();
  solver_parameters->init();
  
  //
  // Physical models:
  //  - Source localization
  //    - Solver::SL_subtraction
  //    - Solver::SL_direct
  //  - Transcranial current stimulation
  //    - Solver::tCS_tDCS
  //    - Solver::tCS_tACS
  //  - Local conductivity estimation
  //    - Solver::tCS_tDCS_local_conductivity
  //
  // export OMP_NUM_THREADS=2
  Solver::Model_solver< /* physical model */ Solver::tCS_tDCS,
			/*solver_parameters->get_number_of_threads_()*/ 2 >  model;

  //
  //
  std::cout << "Loop over solvers" << std::endl;
  model.solver_loop();

  //
  //
  return EXIT_SUCCESS;
}
