#include "Downhill_simplex.h"
// 
// 
// 
Utils::Minimizers::Downhill_simplex::Downhill_simplex():
  It_minimizer()
{}
// 
// 
// 
void
Utils::Minimizers::Downhill_simplex::minimize()
{
  std::cout << "La vie est belle" << std::endl;
}
// 
// 
// 
void
Utils::Minimizers::Downhill_simplex::initialization( Function Func, 
						     const std::vector< Estimation_tuple >& Simplex,
						     const std::map< Brain_segmentation, std::tuple<double, double> >& Boundaries )
{
  function_                = Func;
  simplex_                 = Simplex;
  conductivity_boundaries_ = Boundaries;

  for( auto vertex : simplex_ )
    std::cout << std::get<0>(vertex) << std::endl;

  std::cout << function_(std::get<1>(simplex_[0])) << std::endl;

 }
