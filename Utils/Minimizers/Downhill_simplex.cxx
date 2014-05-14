#include "Downhill_simplex.h"
// 
// 
// 
Utils::Minimizers::Downhill_simplex::Downhill_simplex():
  It_minimizer(),
  delta_(1.e-5), a_(0.25), b_(1.15), c_(0.35)
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
    x1, x2, 
    center;
  double y1, y2;

  // 
  // Downhill simplex main loop
  while( iteration_++ < max_iterations_ && !is_converged() ) 
    {
      std::cout << "Iteration: " << iteration_ << " ~ "
		<< (std::get<1>(simplex_[0]))[0] <<  " ~ "
		<< (std::get<1>(simplex_[0]))[1] <<  " ~ "
		<< (std::get<1>(simplex_[0]))[2] 
		<< std::endl;
      // Order the simplex vertices
      order_vertices();
      // Compute the center of the lightest facet
      center = get_facet_centroid( std::get<1>(simplex_[asc_[0]]), 
				   std::get<1>(simplex_[asc_[1]]), 
				   std::get<1>(simplex_[asc_[2]]) );
      // 
      x1 = center * (1 + a_) - std::get<1>(simplex_[asc_[3]]) * a_;
      y1 = function_(x1);
      // 
      if ( y1 < std::get<0>(simplex_[asc_[0]]) ) 
	{
	  x2 = x1 * (1 + b_) - center * b_;
	  // 
	  if ( (y2 = function_(x2)) < std::get<0>(simplex_[asc_[0]]) ) 
	    {
	      std::get<1>(simplex_[asc_[3]]) = x2;
	      std::get<0>(simplex_[asc_[3]]) = y2;
	    }
	  else 
	    {
	      std::get<1>(simplex_[asc_[3]]) = x1;
	      std::get<0>(simplex_[asc_[3]]) = y1;
	    }
	} 
      else if ( y1 > std::get<0>(simplex_[asc_[1]]) ) 
	{
	  if ( y1 <= std::get<0>(simplex_[asc_[3]]) ) 
	    {
	      std::get<1>(simplex_[asc_[3]]) = x1;
	      std::get<0>(simplex_[asc_[3]]) = y1;
	    }
	  // 
	  x2 = std::get<1>(simplex_[asc_[3]]) * c_ + center * (1 - c_);
	  // 
	  if ( (y2 = function_(x2)) > std::get<0>(simplex_[asc_[3]]) ) 
	    contraction();
	  else 
	    {
	      std::get<1>(simplex_[asc_[3]]) = x2;
	      std::get<0>(simplex_[asc_[3]]) = y2;
	    }
	} 
      else 
	{
	  std::get<1>(simplex_[asc_[3]]) = x1;
	  std::get<0>(simplex_[asc_[3]]) = y1;
	}
    }
  
  // 
  // 
  // return std::get<1>(simplex_[asc_[0]]);
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
//
//  for( auto vertex : simplex_ )
//    std::cout << std::get<0>(vertex) << std::endl;
//
//  std::cout << function_(std::get<1>(simplex_[0])) << std::endl;
//
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
  line[1] = std::get<1>(simplex_[0]) - std::get<1>(simplex_[2]);
  line[2] = std::get<1>(simplex_[0]) - std::get<1>(simplex_[3]);
  line[3] = std::get<1>(simplex_[2]) - std::get<1>(simplex_[1]);
  line[4] = std::get<1>(simplex_[3]) - std::get<1>(simplex_[2]);
  line[5] = std::get<1>(simplex_[1]) - std::get<1>(simplex_[3]);

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
  asc_.clear();
  //TODO  eval();
  for ( int i = 0; i < 4; ++i) 
    {
      if ( asc_.size() == 0 ) asc_.push_back(i);
      else 
	{
	  auto best = asc_.begin();
	  for ( auto it = asc_.begin(); it != asc_.end(); ++it) 
	    {
	      if ( std::get<0>(simplex_[i]) < std::get<0>(simplex_[*it]) )
		best = it;
	      else if ( std::get<0>(simplex_[i]) > std::get<0>(simplex_[*it]) )
		++best;
	    }
	  // 
	  asc_.insert(best, i);
	}
    }
  //	for (vector<int>::iterator it = asc_.begin(); it != asc_.end(); ++it) {
  //		cout << "X_" << *it << " => " << std::get<0>(simplex_[*it]) << endl;
  //	}
  //	cout << endl;
}
// 
// 
// 
const Eigen::Vector3d
Utils::Minimizers::Downhill_simplex::get_facet_centroid( const Eigen::Vector3d& X, 
							 const Eigen::Vector3d& Y, 
							 const Eigen::Vector3d& Z ) const
{
  return ( X + Y + Z ) / 3.;
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
  order_vertices();
  // 
  for ( int i = 1 ; i < 4 ; ++i )
    {
      std::get<1>(simplex_[asc_[i]]) = get_middle( std::get<1>(simplex_[asc_[0]]), 
						   std::get<1>(simplex_[asc_[i]]) );
      // 
      std::get<0>(simplex_[asc_[i]]) = function_( std::get<1>(simplex_[asc_[i]]) );
    }
}
// 
// 
// 
Eigen::Vector3d
Utils::Minimizers::Downhill_simplex::reflection()
{
  // 
  // 
  Eigen::Vector3d center = get_facet_centroid( std::get<1>(simplex_[asc_[0]]), 
					       std::get<1>(simplex_[asc_[1]]), 
					       std::get<1>(simplex_[asc_[2]]) );

  std::cout << std::get<1>(simplex_[asc_[3]]) << std::endl;

  center = center * (1 + a_) - std::get<1>(simplex_[asc_[3]]) * a_;

  std::cout << center << std::endl;
  
  // 
  // 
  return center;
  //	int updatedID = asc_[3];
  //	order();
  //	if (asc_[0] == updatedID)
  //		std::get<1>(simplex_[asc_[0]]) = moveToward(std::get<1>(simplex_[updatedID]), std::get<1>(simplex_[asc_[0]])) * 1.5;
  //	else if (asc_[3] == updatedID)
  //		std::get<1>(simplex_[asc_[3]]) = getMid(std::get<1>(simplex_[asc_[3]]), center);
  //
  //	for (int i = 0; i < 4; ++i) {
  //		cout << std::get<1>(simplex_[i]) << "\n\n";
  //	}
}
// 
// 
// 
// 
// 
// 
