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
#ifndef KMEANS_CLUSTERING_H
#define KMEANS_CLUSTERING_H
#include <random>
//
// UCSF project
//
#include "Fijee/Fijee_log_management.h"
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Minimizers
  {
    namespace Clustering
    {
      /*
       * \brief Lloyd algorythm for k-means clustering
       *
       * This fonction compute the k-means clustering following Lloyd algorithm. 
       * Note: the algorithm can be optimized and transfer on GPU. 
       *
       */
      template < typename T, typename U > void Lloyd_algorythm( const int N, const T* Points, int* Clusters,
								const int n, double* Mu,
								double Epsilon = 1.e-01 )
	{
	  // 
	  // Time log
	  FIJEE_TIME_PROFILER("Utils::Minimizers::Kmeans_clustering::Lloyd_algorythm");

	  // 
	  // Convergence criteria
	  double L_kmeans_old = 1.e+06;
	  double L_kmeans     = 0.;
	  int    iteration    = 0;
	  
	  // 
	  // Determin Max and Min from Points
	  U 
	    max = 0,
	    min = 255;
	  // 
	  for ( int i = 0 ; i < N ; i++ )
	    if( Points[i] > max ) max = static_cast<U>( Points[i] );
	    else if( Points[i] < min ) min = static_cast<U>( Points[i] );
	
	  // 
	  // Uniformly random generation of n of type T between [Min, Max] (Mu)
	  std::default_random_engine generator;
	  std::uniform_int_distribution<U> distribution( min, max );
	  // 
	  for( int i = 0 ; i < n ; i++ ){
	    Mu[i] = distribution( generator );
	    std::cout << Mu[i] << " ";}
	  std::cout << std::endl;

	  // 
	  // Minimization loop
	  while( fabs(L_kmeans - L_kmeans_old) > Epsilon )
	    {
	      // 
	      // 
	      iteration++;
	      // 
	      L_kmeans_old = L_kmeans;
	      L_kmeans = 0;

	      // 
	      // Determin r_ik (Clusters)
	      for( int i = 0 ; i < N ; i++ )
		{
		  double min_dist = 1.e+09;
		  // 
		  for ( int j = 0 ; j < n ; j++)
		    if( fabs(Points[i] - Mu[j]) < min_dist )
		      {
			Clusters[i] = j;
			min_dist = fabs( Points[i] - Mu[j] );
		      }
		}

	      // 
	      // Recompute Mu
	      double 
		Num = 0., 
		Den = 0.;
	      for( int j = 0 ; j < n ; j++ )
		{
		  Num = 0;
		  Den = 0;
		  //	    
		  for( int i = 0 ; i < N ; i++ )
		    if( Clusters[i] == j )
		      {
			Num += Points[i];
			Den++;
		      }
		  //
		  Mu[j]  = static_cast<double>( Num );
		  Mu[j] /= static_cast<double>( Den );
		}

	      // 
	      // Compute the Lagrangien
	      for( int j = 0 ; j < n ; j++ )
		for( int i = 0 ; i < N ; i++ )
		  if( Clusters[i] == j )
		    L_kmeans += (Points[i]-Mu[j])*(Points[i]-Mu[j]);
	    }


	  //
	  //
	  std::cout << std::endl;
	  std::cout << iteration << std::endl;
	  for( int j = 0 ; j < n ; j++ ){
	    std::cout << Mu[j] << std::endl;}
	  std::cout << std::endl;
	}
    }
  }
}
#endif
