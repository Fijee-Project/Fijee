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
#include <cstdlib>
//
// UCSF
//
#include "Noise.h"
#include "Empirical_bayesian_noise.h"
#include "Inverse_parameters.h"
#include "Inverse_model.h"
#include "Inverse_solver.h"
#include "Empirical_bayesian_model.h"
//
typedef Inverse::Empirical_bayesian_model< Inverse::Empirical_bayesian_noise > Inverser_algorithm;
//
int main()
{
  //
  //
  Inverse::Inverse_parameters* inverse_parameters = Inverse::Inverse_parameters::get_instance();
  inverse_parameters->init();
  
  //
  // Inverse models:
  //  - Empirical bayesian model
  //    - Inverse::Empirical_bayesian_model
  //
  // export OMP_NUM_THREADS=2
  Inverse::Inverse_model< /* inverse model */ Inverser_algorithm,
			  /*inverse_parameters->get_number_of_threads_()*/ 2 >  model;
//
//  //
//  //
//  //  std::cout << "Loop over solvers" << std::endl;
//  model.solver_loop();

  //
  //
  return EXIT_SUCCESS;
}
