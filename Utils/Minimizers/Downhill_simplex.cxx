#include "Downhill_simplex.h"
// 
// 
// 
Utils::Minimizers::Downhill_simplex::Downhill_simplex():
  It_minimizer(),
  delta_(1.e-5), reflection_coeff_(0.25), expension_coeff_(1.15), contraction_coeff_(0.35)
{}
// 
// 
// 
void
Utils::Minimizers::Downhill_simplex::minimize()
{
  // 
  // 
  Eigen::Vector3d 
    x_reflection, 
    x_expension, 
    x_contraction, 
    center;
  double 
    y_reflection, 
    y_expension,
    y_contraction;

  // 
  // Downhill simplex main loop
  while( iteration_++ < max_iterations_ && !is_converged() ) 
    {
      std::cout << "Iteration: " << iteration_ 
		<< " ~0~ " << std::get<0>(simplex_[0]) <<  " | "
		<< (std::get<1>(simplex_[0]))[0] <<  ", "
		<< (std::get<1>(simplex_[0]))[1] <<  ", "
		<< (std::get<1>(simplex_[0]))[2] 
		<< " ~1~ " << std::get<0>(simplex_[1]) <<  " | "
		<< (std::get<1>(simplex_[1]))[0] <<  ", "
		<< (std::get<1>(simplex_[1]))[1] <<  ", "
		<< (std::get<1>(simplex_[1]))[2] 
		<< " ~2~ " << std::get<0>(simplex_[N_]) <<  " | "
		<< (std::get<1>(simplex_[N_]))[0] <<  ", "
		<< (std::get<1>(simplex_[N_]))[1] <<  ", "
		<< (std::get<1>(simplex_[N_]))[2] 
		<< " ~3~ " << std::get<0>(simplex_[N_+1]) <<  " | "
		<< (std::get<1>(simplex_[N_+1]))[0] <<  ", "
		<< (std::get<1>(simplex_[N_+1]))[1] <<  ", "
		<< (std::get<1>(simplex_[N_+1]))[2] 
		<< std::endl;
      //
      // Order the simplex vertices
      //      order_vertices();
      std::sort(simplex_.begin(), simplex_.end(), []( const Estimation_tuple& tuple1,
						      const Estimation_tuple& tuple2 )
		 ->bool
		 {
		   return ( std::get<0>(tuple1) < std::get<0>(tuple2) ); 
		 });

      std::cout << "Iteration: " << iteration_ 
		<< " ~0~ " << std::get<0>(simplex_[0]) <<  " | "
		<< (std::get<1>(simplex_[0]))[0] <<  ", "
		<< (std::get<1>(simplex_[0]))[1] <<  ", "
		<< (std::get<1>(simplex_[0]))[2] 
		<< " ~1~ " << std::get<0>(simplex_[1]) <<  " | "
		<< (std::get<1>(simplex_[1]))[0] <<  ", "
		<< (std::get<1>(simplex_[1]))[1] <<  ", "
		<< (std::get<1>(simplex_[1]))[2] 
		<< " ~2~ " << std::get<0>(simplex_[N_]) <<  " | "
		<< (std::get<1>(simplex_[N_]))[0] <<  ", "
		<< (std::get<1>(simplex_[N_]))[1] <<  ", "
		<< (std::get<1>(simplex_[N_]))[2] 
		<< " ~3~ " << std::get<0>(simplex_[N_+1]) <<  " | "
		<< (std::get<1>(simplex_[N_+1]))[0] <<  ", "
		<< (std::get<1>(simplex_[N_+1]))[1] <<  ", "
		<< (std::get<1>(simplex_[N_+1]))[2] 
		<< std::endl;
      // Compute the center of the lightest facet
      center = get_facet_centroid();
      // 
      x_reflection = center*(1 + reflection_coeff_) - std::get<1>(simplex_[N_+1])*reflection_coeff_;
      y_reflection = function_(x_reflection);
      // 
      if( y_reflection >= std::get<0>(simplex_[0]) &&  
	  y_reflection <= std::get<0>(simplex_[N_]) )
	{
	  ////////////////
	  // Reflection //
	  ////////////////
	  std::get<1>(simplex_[N_+1]) = x_reflection;
	  std::get<0>(simplex_[N_+1]) = y_reflection;
	}
      else if ( y_reflection < std::get<0>(simplex_[0]) ) 
	{
	  ///////////////
	  // Expension //
	  ///////////////
	  x_expension = x_reflection * (1 + expension_coeff_) - center * expension_coeff_;
	  // 
	  // TODO if ( (y_expension = function_(x_expension)) < std::get<0>(simplex_[0]) ) 
	  if ( (y_expension = function_(x_expension)) < y_reflection ) 
	    {
	      std::get<1>(simplex_[N_+1]) = x_expension;
	      std::get<0>(simplex_[N_+1]) = y_expension;
	    }
	  else 
	    {
	      std::get<1>(simplex_[N_+1]) = x_reflection;
	      std::get<0>(simplex_[N_+1]) = y_reflection;
	    }
	} 
      else if ( y_reflection > std::get<0>(simplex_[N_]) ) 
	{
	  /////////////////
	  // Contraction //
	  /////////////////
	  x_contraction = std::get<1>(simplex_[N_+1])*contraction_coeff_ + center*(1 - contraction_coeff_);
	  //
	  if ( (y_contraction = function_(x_contraction)) <= std::get<0>(simplex_[N_+1]) ) 
	    {
	      std::get<1>(simplex_[N_+1]) = x_contraction;
	      std::get<0>(simplex_[N_+1]) = y_contraction;
	    }
	  else
	    {
	      // Multiple contraction
	      contraction();
	    }
	}
      else 
	{
	  std::cerr << "Downhill simplex: case unknown" << std::endl;
	  abort();
	}
    }
  // 
  // 
  // return std::get<1>(simplex_[0]);
}
// 
// 
// 
void
Utils::Minimizers::Downhill_simplex::initialization( Function Func, 
						     const std::vector< Estimation_tuple >& Simplex,
						     const std::vector< std::tuple<double, double> >& Boundaries )
{
  function_                = Func;
  simplex_                 = Simplex;
  conductivity_boundaries_ = Boundaries;
  // N+1 dimension of the simplex
  // N dimension of the spcace: [0, 1, ..., (N-1)]
  N_ = (simplex_.size() - 1) - 1;
 }
// 
// 
// 
bool 
Utils::Minimizers::Downhill_simplex::is_converged()
{
  // calculate all lines length
  Eigen::Vector3d line[6];
  line[0] = std::get<1>(simplex_[0]) - std::get<1>(simplex_[1]);
  line[1] = std::get<1>(simplex_[0]) - std::get<1>(simplex_[N_]);
  line[2] = std::get<1>(simplex_[0]) - std::get<1>(simplex_[N_+1]);
  line[3] = std::get<1>(simplex_[N_]) - std::get<1>(simplex_[1]);
  line[4] = std::get<1>(simplex_[N_+1]) - std::get<1>(simplex_[N_]);
  line[5] = std::get<1>(simplex_[1]) - std::get<1>(simplex_[N_+1]);

  // 
  return (line[0].norm() < delta_) && (line[1].norm() < delta_)
    && (line[2].norm() < delta_) && (line[3].norm() < delta_)
    && (line[4].norm() < delta_) && (line[5].norm() < delta_);
}
// 
// 
// 
void
Utils::Minimizers::Downhill_simplex::order_vertices()
{
  // Obselete
//  asc_.clear();
//  //TODO  eval();
//  for ( int i = 0; i < 4; ++i) 
//    {
//      if ( asc_.size() == 0 ) asc_.push_back(i);
//      else 
//	{
//	  auto best = asc_.begin();
//	  for ( auto it = asc_.begin(); it != asc_.end(); ++it) 
//	    {
//	      if ( std::get<0>(simplex_[i]) < std::get<0>(simplex_[*it]) )
//		best = it;
//	      else if ( std::get<0>(simplex_[i]) > std::get<0>(simplex_[*it]) )
//		++best;
//	    }
//	  // 
//	  asc_.insert(best, i);
//	}
//    }
//  //	for (vector<int>::iterator it = asc_.begin(); it != asc_.end(); ++it) {
//  //		cout << "X_" << *it << " => " << std::get<0>(simplex_[*it]) << endl;
//  //	}
//  //	cout << endl;
}
// 
// 
// 
const Eigen::Vector3d
Utils::Minimizers::Downhill_simplex::get_facet_centroid() const
{
  Eigen::Vector3d Sum;
  for ( int i = 0 ; i < (N_+1) ; i++)
    Sum += std::get<1>(simplex_[i]) ;
  
  // 
  // 
  return Sum / 3.;
}
// 
// 
// 
const Eigen::Vector3d 
Utils::Minimizers::Downhill_simplex::get_middle( const Eigen::Vector3d& X, 
						 const Eigen::Vector3d& Y ) const
{
  return ( X + Y ) / 2.;
}
// 
// 
//
void
Utils::Minimizers::Downhill_simplex::contraction()
{
  // order_vertices();
  std::sort(simplex_.begin(), simplex_.end(), []( const Estimation_tuple& tuple1,
						  const Estimation_tuple& tuple2 )
	    ->bool
	    {
	      return ( std::get<0>(tuple1) < std::get<0>(tuple2) ); 
	    });
  // Move vertices toward the lighter vertice: simplex_[0]
  for ( int i = 1 ; i < (N_+1) + 1 ; ++i )
    {
      std::get<1>(simplex_[i]) = get_middle( std::get<1>(simplex_[0]), 
					     std::get<1>(simplex_[i]) );
      // 
      std::get<0>(simplex_[i]) = function_( std::get<1>(simplex_[i]) );
    }
}
// 
// 
// 
Eigen::Vector3d
Utils::Minimizers::Downhill_simplex::reflection()
{
  // obselete
//  // 
//  // 
//  Eigen::Vector3d center = get_facet_centroid( std::get<1>(simplex_[0]), 
//					       std::get<1>(simplex_[1]), 
//					       std::get<1>(simplex_[N_]) );
//
//  std::cout << std::get<1>(simplex_[N_+1]) << std::endl;
//
//  center = center * (1 + reflection_coeff_) - std::get<1>(simplex_[N_+1]) * reflection_coeff_;
//
//  std::cout << center << std::endl;
//  
//  // 
//  // 
//  return center;
//  //	int updatedID = 3;
//  //	order();
//  //	if (0 == updatedID)
//  //		std::get<1>(simplex_[0]) = moveToward(std::get<1>(simplex_[updatedID]), std::get<1>(simplex_[0])) * 1.5;
//  //	else if (3 == updatedID)
//  //		std::get<1>(simplex_[3]) = getMid(std::get<1>(simplex_[3]), center);
//  //
//  //	for (int i = 0; i < 4; ++i) {
//  //		cout << std::get<1>(simplex_[i]) << "\n\n";
//  //	}
}
