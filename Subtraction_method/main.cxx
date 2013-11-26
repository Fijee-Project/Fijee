#include <vector>
#include <memory>
//
// UCSF
//
#include "PDE_solver_parameters.h"
#include "Physical_model.h"
#include "Subtraction.h"

int main()
{
  //
  //
  Solver::PDE_solver_parameters* solver_parameters = Solver::PDE_solver_parameters::get_instance();
  solver_parameters->init();

  //
  // 
  std::unique_ptr< Solver::Physical_model> model( new Solver::Subtraction() );

  //
  //
  std::cout << "Loop over solvers" << std::endl;
  model->solver_loop();

  //
  //
  return EXIT_SUCCESS;
}
