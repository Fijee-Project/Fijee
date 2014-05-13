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
  // physical models:
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
  Solver::Model_solver< /* physical model */ Solver::tCS_tDCS_local_conductivity,
			/*solver_parameters->get_number_of_threads_()*/ 2 >  model;

  //
  //
  std::cout << "Loop over solvers" << std::endl;
  model.solver_loop();

  //
  //
  return EXIT_SUCCESS;
}
