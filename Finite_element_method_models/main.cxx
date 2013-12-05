#include <vector>
#include <memory>
//
// UCSF
//
#include "PDE_solver_parameters.h"
#include "SL_subtraction.h"
#include "SL_direct.h"
#include "Model_solver.h"

int main()
{
  //
  //
  Solver::PDE_solver_parameters* solver_parameters = Solver::PDE_solver_parameters::get_instance();
  solver_parameters->init();
  
  //
  // physical models:
  //  - Solver::SL_subtraction
  //  - Solver::SL_direct
  Solver::Model_solver< /* physical model */ Solver::SL_subtraction,
			/*solver_parameters->get_number_of_threads_()*/ 2>  model;

  //
  //
  std::cout << "Loop over solvers" << std::endl;
  model.solver_loop();

  //
  //
  return EXIT_SUCCESS;
}
