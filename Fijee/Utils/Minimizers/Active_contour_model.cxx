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
#include "Active_contour_model.h"
// 
// 
// 
Utils::Minimizers::Active_contour_model::Active_contour_model():
  It_minimizer(),
  snakelength_(0.), width_(0), height_(0), slice_z_(0),
  gradient_(nullptr), flow_(nullptr), 
  alpha_(0.), beta_(0.), gamma_(0.), delta_(0.), 
  Max_iteration_(0)
{}
// 
// 
// 
Utils::Minimizers::Active_contour_model::Active_contour_model( const int Number_of_pixels_x, 
							       const int Number_of_pixels_y, 
							       const int K,
							       const int* Image,
							       const double Threshold,
							       bool Sobel ):
  It_minimizer(),
  snakelength_(0.), width_(Number_of_pixels_x), height_(Number_of_pixels_y),  slice_z_(K),
  gradient_(nullptr), flow_(nullptr), 
  alpha_(1.1), beta_(1.2), gamma_(1.5), delta_(3.0), 
  Max_iteration_(1000)
{
  // 
  // Build the snake
  for( int j = 1 ; j < Number_of_pixels_y-1 ; j++)
    snake_.push_back(BPoint( 1, j, K ));
  for( int i = 1 ; i < Number_of_pixels_x-1 ; i++)
    snake_.push_back(BPoint( i, Number_of_pixels_y-2, K ));
  for( int j = 0 ; j < Number_of_pixels_y-2 ; j++)
    snake_.push_back(BPoint( Number_of_pixels_x-2, Number_of_pixels_y-2 - j, K ));
//  for( int i = 0 ; i < Number_of_pixels_x - 1 ; i++)
//    snake_.push_back(BPoint( Number_of_pixels_x-1 - i, 0, K ));
  
  // 
  // Build the gradiant map
  if( Sobel )
    image_gradient( Image );
  else
    Sobel_filter( Image );

  // 
  // Build the flow map
  image_flow( Threshold );
}
// 
// 
// 
Utils::Minimizers::Active_contour_model::~Active_contour_model()
{
  // 
  // 
  if( gradient_ )
    {
      delete[] gradient_;
      gradient_ = nullptr;
    }
  // 
  if( flow_ )
    {
      delete[] flow_;
      flow_ = nullptr;
    }
};
// 
// 
// 
void
Utils::Minimizers::Active_contour_model::minimize()
{
  // 
  // 
  int iteration = 0;
  
  while( iteration_step() && iteration < Max_iteration_ ) 
    {
      // 
      // auto adapt the number of points in the snake
      if ( iteration%10 == 0 ) 
	{
	  remove_overlapping_points();
	  add_missing_points();
	}
      // 
      iteration++;
    }
  
  // 
  // rebuild using spline interpolation
  rebuild();
}
// 
// 
// 
void
Utils::Minimizers::Active_contour_model::image_gradient( const int* Image )
{
  // 
  // 
  double max_gradient_amplitude = 1.;
  // 
  gradient_ = new double[width_*height_];

  // 
  // 
  for ( int j = 1 ; j < height_ - 1 ; j++ )
    for ( int i = 1 ; i < width_ - 1 ; i++ )
      {
	double
	  partial_x = Image[ i+1 + width_*j + width_*height_*slice_z_ ] - Image[ i-1 + width_*j + width_*height_*slice_z_ ],
	  partial_y = Image[ i + width_*(j+1) + width_*height_*slice_z_ ] - Image[ i + width_*(j-1) + width_*height_*slice_z_ ];
	// 
	gradient_[ i + width_*j ] = partial_x*partial_x + partial_y*partial_y;
	// normalization terme
	if( gradient_[ i + width_*j ] > max_gradient_amplitude )
	  max_gradient_amplitude = gradient_[ i + width_*j ];
      }

  // 
  // Normalization of the gradient
  for( int i = 0 ; i < height_*width_ ; i++ )
    gradient_[i] /= max_gradient_amplitude;
}
// 
// 
// 
void
Utils::Minimizers::Active_contour_model::image_flow( const double Threshold )
{
  // 
  // Binarization of the gradient
  flow_ = new double[width_*height_];
  // 
  for( int i = 0 ; i < height_*width_ ; i++ )
    flow_[i] = ( gradient_[i] > Threshold ? 0 : -1 );

  // 
  // 
  int chamfer[2][3] = {{1,0,3},{1,1,4}};
  double normalizer = chamfer[0][2];


  // 
  // Code from Xavier Philippeau
  // http://www.developpez.net/forums/d474799/general-developpement-1/algorithme-mathematiques-2031/contribuez-416/java-carte-distances-chamfer/
  // 

  // 
  // Forward
  for (int j = 0 ; j < height_ ; j++)
    for (int i = 0 ; i < width_ ; i++) 
      {
	// 
	// 
	double value = flow_[ i + height_*j ];
	// 
	if ( value < 0 )
	  for(int k = 0 ; k < 2 ; k++ ) 
	    {
	      int di = chamfer[k][0];
	      int dj = chamfer[k][1];
	      int dt = chamfer[k][2];
	      // 
	      set_value(flow_, i+di, j+dj, value+dt);
	      //
	      if ( dj != 0 ) 
		set_value( flow_, i-di, j+dj, value+dt);
	      // 
	      if (di != dj) 
		{
		  set_value(flow_, i+dj, j+di, value+dt);
		  // 
		  if ( dj != 0) 
		    set_value(flow_, i-dj, j+di, value+dt);
		}
	    }
      }
 

  // 
  // Backward
  for( int j = height_-1 ; j >= 0 ; j-- ) 
    for( int i = width_-1; i >= 0 ; i-- ) 
      {
	// 
	// 
	double value = flow_[i + j*height_];
	// 
	if( value < 0 )
	  for( int k = 0 ; k < 2 ; k++ )
	    {
	      int di = chamfer[k][0];
	      int dj = chamfer[k][1];
	      int dt = chamfer[k][2];
	      
	      set_value(flow_, i-di, j-dj, value+dt);
	      if (dj!=0) set_value(flow_, i+di, j-dj, value+dt);
	      if (di!=dj) {
		set_value(flow_, i-dj, j-di, value+dt);
		if (dj!=0) set_value(flow_, i+dj, j-di, value+dt);
	      }
	    }
      }
  

  //
  // normalize
  for(int i = 0; i < width_*height_ ; i++ )
    flow_[i] /= normalizer;
}
// 
// 
// 
bool
Utils::Minimizers::Active_contour_model::iteration_step()
{
  //
  //
  bool changed = false;
  BPoint p( 0, 0, slice_z_ );
 
  // compute length of original snake (used by method: f_uniformity)
  get_snakelength_();
 
  // compute the new snake
  std::list<BPoint> newsnake;
 
  // 
  // for each point of the previous snake
  auto point = snake_.begin(); point++;
  // 
  for( ; point != std::next(snake_.end(),-1) ; point++ ) 
    {
      // 
      // 
      BPoint prev = *(std::next(point,-1));
      BPoint cur  = *(point);
      BPoint next = *(std::next(point,+1));
      // 3x3 neighborhood used to compute energies
      // - 0: e_uniformity: internal energy
      // - 1: e_curvature:  internal energy
      // - 2: e_flow: external energy
      // - 3: e_inertia: external energy
      double energy[4][3][3];
      double e_norm[4];
      
      double e_uniformity[3][3];
      double e_curvature[3][3];
      double e_flow[3][3];
      double e_inertia[3][3];

      
      // 
      // compute all energies
      for( int dy = -1 ; dy <= 1 ; dy++ ) 
	for( int dx = -1 ; dx <= 1 ; dx++ ) 
	  {
	    // 
	    // 
	    p = BPoint( cur.x()+dx, cur.y()+dy, slice_z_ );
	    // 
	    energy[0][1+dx][1+dy] = f_uniformity( prev, next, p );
	    energy[1][1+dx][1+dy] = f_curvature( prev, p, next );
	    energy[2][1+dx][1+dy] = f_gflow( cur, p );
	    energy[3][1+dx][1+dy] = f_inertia( cur, p );
	    //
	    for( int e = 0 ; e < 4 ; e++)
	      e_norm[e] += energy[e][1+dx][1+dy];
	  }
      // Normalization
      for( int e = 0 ; e < 4 ; e++ ) 
	for( int i = 0 ; i < 3 ; i++ ) 
	  for( int j = 0 ; j < 3 ; j++ ) 
	    if( e_norm[e] != 0 )
	      energy[e][i][j] /= e_norm[e];

      
      // 
      // find the point with the minimum sum of energies
      double emin = 1.e+09;
      int 
	x = 0,
	y = 0;
      for( int dy = -1 ; dy <= 1 ; dy++ ) 
	for( int dx = -1 ; dx <= 1 ; dx++ ) 
	  {
	    double energy_tempo = 0.;
	    // 
	    energy_tempo += alpha_ * energy[0][1+dx][1+dy];
	    energy_tempo += beta_  * energy[1][1+dx][1+dy];
	    energy_tempo += gamma_ * energy[2][1+dx][1+dy];
	    energy_tempo += delta_ * energy[3][1+dx][1+dy];
	    
	    if ( energy_tempo < emin ) 
	      { emin = energy_tempo; 
		x = cur.x()+dx; 
		y = cur.y()+dy; 
	      }
	  }
 
      // 
      // boundary check
      if ( x < 1 ) x = 1;
      if ( x >= width_-1 ) x = width_ - 2;
      if ( y < 1 ) y = 1;
      if (y >= height_-1 ) y = height_ - 2;
      
      // compute the returned value
      if ( x != cur.x() || y != cur.y() ) 
	changed = true;
      
      // create the point in the new snake
      newsnake.push_back( BPoint(x,y, slice_z_) );
    }
  
  // new snake becomes current
  snake_.swap( newsnake );
  
  // 
  //
  return changed;
}
//
//
//
void
Utils::Minimizers::Active_contour_model::remove_overlapping_points()
{
  // 
  // 
  for( auto point = snake_.begin() ; point != snake_.end() ; point++ )
    for ( auto next = (std::next(point,1)) ; next != snake_.end() ; next++ )
      if( point->squared_distance(*next) < 2*2 )
	next = snake_.erase(next); 
}
//
//
//
void
Utils::Minimizers::Active_contour_model::add_missing_points()
{
  // 
  // 
  auto point = snake_.begin(); point++;
  // for each point of the snake
  for( ; point != snake_.end() ; point++ )
    if( (std::next(point,1)) != snake_.end() )
      if( (std::next(point,2)) != snake_.end() )
	{
	  BPoint prev   = *(std::next(point,-1));
	  BPoint cur    = *(point);
	  BPoint next   = *(std::next(point,+1));
	  BPoint next2  = *(std::next(point,+2));
	  
	  // if the next point is to far then add a new point
	  if ( cur.squared_distance(next) > 8*8 ) 
	    {
	      // 
	      // precomputed Uniform cubic B-spline for t=0.5
	      double 
		c0 = 0.125/6.0, 
		c1 = 2.875/6.0, 
		c2 = 2.875/6.0, 
		c3 = 0.125/6.0;
	      // 
	      double 
		x = prev.x()*c3 + cur.x()*c2 + next.x()* c1 + next2.x()*c0,
		y = prev.y()*c3 + cur.y()*c2 + next.y()* c1 + next2.y()*c0;
	      BPoint newpoint( static_cast<int>(0.5+x), static_cast<int>(0.5+y), slice_z_ );
	      
	      snake_.insert( (std::next(point,1)) , newpoint ); point--; // ??
	    }
	}
}
//
//
//
void
Utils::Minimizers::Active_contour_model::rebuild()
{
  //
  // precompute length(i) = length of the snake from start to point #i
  double clength[ snake_.size() + 1 ];
  //
  auto point = snake_.begin();
  for( int i = 0 ; point != snake_.end() ; point++, i++ ) 
    if( (std::next(point,1)) != snake_.end() )
    {
      BPoint cur   = *(point);
      BPoint next  = *(std::next(point,1));
      // 
      clength[i+1] = clength[i] + sqrt(cur.squared_distance(next));
    }
 
  // 
  // compute number of points in the new snake
  double total = clength[snake_.size()];
  int nmb      = static_cast<int>( 0.5 + total / 16. );
 
  // 
  // build a new snake
  std::list< BPoint > newsnake;
  // 
  point = snake_.begin();
  int i = 0;
  for( int j = 0 ; j < nmb ; j++ ) 
    if( (std::next(point,1)) != snake_.end() )
      if( (std::next(point,2)) != snake_.end() )
	{
	  // 
	  // current length in the new snake
	  double dist = (j*total)/nmb;
	  
	  // find corresponding interval of points in the original snake
	  while( !(clength[i] <= dist && dist < clength[i+1]) )
	    {
	      i++; point++;
	    }
	  
	  // get points (P-1,P,P+1,P+2) in the original snake
	  BPoint prev   = *(std::next(point,-1));
	  BPoint cur    = *(point);
	  BPoint next   = *(std::next(point,+1));
	  BPoint next2  = *(std::next(point,+2));
      
	  // do cubic spline interpolation
	  double t =   (dist-clength[i])/(clength[i+1]-clength[i]);
	  double t2 =  t*t, t3=t2*t;
	  double c0 =  1*t3;
	  double c1 = -3*t3 +3*t2 +3*t + 1;
	  double c2 =  3*t3 -6*t2 + 4;
	  double c3 = -1*t3 +3*t2 -3*t + 1;
	  double x  =  prev.x()*c3 + cur.x()*c2 + next.x()* c1 + next2.x()*c0;
	  double y  =  prev.y()*c3 + cur.y()*c2 + next.y()* c1 + next2.y()*c0;
	  BPoint newpoint( static_cast<int>(0.5+x/6), static_cast<int>(0.5+y/6), slice_z_ );
      
	  // add computed point to the new snake
	  newsnake.push_back( newpoint );
	}

  // 
  // 
  snake_.swap( newsnake );
}
