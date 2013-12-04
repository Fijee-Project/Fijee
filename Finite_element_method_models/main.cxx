#include <vector>
#include <memory>
//
// UCSF
//
#include "PDE_solver_parameters.h"
#include "SL_subtraction.h"
#include "Model_solver.h"

int main()
{
  //
  //
  Solver::PDE_solver_parameters* solver_parameters = Solver::PDE_solver_parameters::get_instance();
  solver_parameters->init();
  
  //
  // 
  Solver::Model_solver< Solver::SL_subtraction /* physical model */,
			2 >  model; // solver_parameters->get_number_of_threads_()

  //
  //
  std::cout << "Loop over solvers" << std::endl;
  model.solver_loop();

  //
  //
  return EXIT_SUCCESS;
}
