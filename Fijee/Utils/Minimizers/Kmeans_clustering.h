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
#include "Utils/Data_structure/Graph_abstract_data_type.h"
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
	
	//	  // 
	//	  std::uniform_int_distribution<U> distribution( min, max );
	// 
	double set_size = max - min;
	double domaine  = set_size / (double) n;
	double min_max_cluster[n][2];
	// positions Mu in the center of domaines
	for( int j = 0 ; j < n ; j++ )
	  {
	    Mu[j] = domaine/2. + j*domaine;
	    std::cout << Mu[j] << " ";}
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
	    // Determins r_ik (Clusters)
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
      /*
       * \brief Fuzzy c-means clustering
       *
       * This fonction compute the fuzzy c-means clustering following accounting the neighbor system.
       * In this algorithm we take the fuzzifier m = 2.
       * Note: the algorithm can be optimized and transfer on GPU. 
       *
       */
      template < typename T, typename U > void Fuzzy_c_means( const int N/*num of sites*/, const T* Points, int* Clusters,
							      const int n/*num od clust*/, double* Mu,
							      const Utils::Data_structure::Graph_abstract_data_type<U> Neighborhood_system, bool Use_ns = false, 
							      double Epsilon = 1.e-01 )
      {
	// 
	// Time log
	FIJEE_TIME_PROFILER("Utils::Minimizers::Kmeans_clustering::Fuzzy_c_means");

	// 
	// Convergence criteria
	double L_cmeans_old = 1.e+06;
	double L_cmeans     = 0.;
	int    iteration    = 0;
	  
	// 
	// Membership matrix
	double* U_ji[n];
	double* H_ji[n];
	//
	double *x = new double[N];


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
	// 
	double set_size = max - min;
	double domaine  = set_size / (double) n;
	double min_max_cluster[n][2];
	// positions Mu in the center of domaines
	for( int j = 0 ; j < n ; j++ )
	  {
	    U_ji[j] = new double[N];
	    H_ji[j] = new double[N];
	    // 
	    Mu[j] = domaine/2. + j*domaine;
	    // 
	    std::cout << Mu[j] << " ";
	  }
	std::cout << std::endl;


	// 
	// Minimization loop
	while( 100 * fabs(L_cmeans-L_cmeans_old) / L_cmeans > Epsilon )
	  {
	    // 
	    // 
	    iteration++;
	    std::cout 
	      << "it: " << iteration << " Cost function L - " 
	      << "L_old: " << L_cmeans_old << " " 
	      << "L_new: " << L_cmeans << " " 
	      << 100 * fabs(L_cmeans-L_cmeans_old) / L_cmeans << "%"
	      << std::endl;

	    // 
	    L_cmeans_old = L_cmeans;
	    L_cmeans = 0;

	    // 
	    // Determins U_ji (Clusters)
	    for( int i = 0 ; i < N ; i++ )
	      {
		double 
		  x_i         = Points[i],
		  Denominator = 0.;
		// 
		for( int k = 0 ; k < n ; k++ )
		  {
		    double deno  = fabs(x_i - Mu[k]);
		    Denominator += 1./(deno*deno);
		  }
		// 
		for( int j = 0 ; j < n ; j++ )
		  {
		    double numerator = fabs(x_i - Mu[j]);
		    // 
		    numerator *= numerator*Denominator;
		    U_ji[j][i] = 1. / numerator;
		  }
	      }

	    if(Use_ns&&iteration>20)
	      {
		// 
		// Determins H_ji
		for( int w = 0 ; w < 256 ; w++ )
		  for( int v = 0 ; v < 256 ; v++ )
		    for( int u = 0 ; u < 256 ; u++ )
		      {
			int i      = u + 256*v + 256*256*w;
			double prob_max = 0.;
			// 
			for( int j = 0 ; j < n ; j++ )
			  {
			    H_ji[j][i] = 1.;
			    for( auto vertex : Neighborhood_system.get_vertices_() )
			      if( u + vertex.x() >= 0 && u + vertex.x() < 256 &&
				  v + vertex.y() >= 0 && v + vertex.y() < 256 &&
				  w + vertex.z() >= 0 && w + vertex.z() < 256 )
				{
				  int k = u+vertex.x() + 256*(v+vertex.y()) + 256*256*(w+vertex.z());
				  H_ji[j][i] *= 256. - fabs(Mu[Clusters[i]]-Mu[Clusters[k]]);
				  H_ji[j][i] /= 256.;
				  //				    H_ji[j][i] += U_ji[j][k];
//				  std::cout << "H_ji["<<j<<"]["<<i<<"]=" <<  H_ji[j][i] 
//					    << " Mu["<<Clusters[i]<<"]= " << Mu[Clusters[i]]
//					    << " Mu["<<Clusters[k]<<"]= " << Mu[Clusters[k]]
//					    << std::endl;
				}
			    // 
			    if( H_ji[j][i] > prob_max )
			      prob_max = H_ji[j][i];
			  }
			// 
			x[i] = 1./prob_max;
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
		  {
		    if(Use_ns&&iteration>20)
		      {
//			if( x[i] != 1 )
//			  std::cout 
//			    << "av U_ji["<<j<<"]["<<i<<"]: " << U_ji[j][i] 
//			    << " H_ji[j][i]=" << H_ji[j][i] 
//			    << " x[i]=" << x[i] << std::endl;
			// 
			// Update U_ji with H_ji
			//			  U_ji[j][i] = U_ji[j][i]*H_ji[j][i];
			U_ji[j][i] *= x[i]*H_ji[j][i];
//			if( x[i] != 1 )
//			  std::cout 
//			    << "ap U_ji["<<j<<"]["<<i<<"]: " << U_ji[j][i] 
//			    << " H_ji[j][i]=" << H_ji[j][i]
//			    << " x[i]=" << x[i] << std::endl;
			double new_denominator = 0.;
			//			  // 
			//			  for( int jj = 0 ; jj < n ; jj++ )
			//			    {
			//			      new_denominator = /*U_ji[jj][i]**/H_ji[jj][i];
			////     std::cout << H_ji[jj][i] << " "  << U_ji[jj][i] << " " << new_U << " " << std::endl;
			//			    }
			//			  
			//			  // 
			//			  U_ji[j][i] /= new_denominator;
			//			  //			  std::cout << "ap U_ji["<<j<<"]["<<i<<"]: " << U_ji[j][i] << std::endl;
		      }

		    // 
		    // 
		    Num += U_ji[j][i]*U_ji[j][i]*static_cast<double>(Points[i]);
		    Den += U_ji[j][i]*U_ji[j][i];
		  }
		//
		Mu[j]  = Num / Den;
	      }

	    // 
	    // Compute the Lagrangien
	    for( int i = 0 ; i < N ; i++ )
	      {
		double cluster_membership = 0.;
		for( int j = 0 ; j < n ; j++ )
		  {
		    L_cmeans += U_ji[j][i]*U_ji[j][i]*(Points[i]-Mu[j])*(Points[i]-Mu[j]);
		    //
		    if( cluster_membership < U_ji[j][i] ) 
		      {
			cluster_membership = U_ji[j][i];
			Clusters[i] = j;
		      }
		  }
	      }
	  }


	//
	//
	std::cout << std::endl;
	std::cout << iteration << std::endl;
	for( int j = 0 ; j < n ; j++ ){
	  std::cout << " " << Mu[j];}
	std::cout << std::endl;


	// 
	// 
	for( int j = 0 ; j < n ; j++ )
	  {
	    delete[] U_ji[j];
	    U_ji[j] = nullptr;
	    delete[] H_ji[j];
	    H_ji[j] = nullptr;
	  }
	// 
	delete[] x;
	x = nullptr;
      }
      /*
       * \brief Fuzzy c-means clustering
       *
       * This fonction compute the fuzzy c-means clustering following accounting the neighbor system.
       * In this algorithm we take the fuzzifier m = 2.
       * Note: the algorithm can be optimized and transfer on GPU. 
       *
       */
      template < typename T, typename U > void Fuzzy_c_means( const int N/*num of sites*/, const T* Points, const U* Bg_mask, U* Clusters, 
							      const int n/*num od clust*/, double* Mu,
							      double Epsilon = 1.e-01 )
      {
	// 
	// Time log
	FIJEE_TIME_PROFILER("Utils::Minimizers::Kmeans_clustering::Fuzzy_c_means");

	// 
	// Convergence criteria
	double L_cmeans_old = 1.e+06;
	double L_cmeans     = 0.;
	int    iteration    = 0;
	  
	// 
	// Membership matrix
	double* U_ji[n];
	//
	double *x = new double[N];


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
	// 
	double set_size = max - min;
	double domaine  = set_size / (double) n;
	double min_max_cluster[n][2];
	// positions Mu in the center of domaines
	for( int j = 0 ; j < n ; j++ )
	  {
	    U_ji[j] = new double[N];
	    // 
	    Mu[j] = domaine/2. + j*domaine;
	    // 
	    std::cout << Mu[j] << " ";
	  }
	std::cout << std::endl;


	// 
	// Minimization loop
	while( 100 * fabs(L_cmeans-L_cmeans_old) / L_cmeans > Epsilon )
	  {
	    // 
	    // 
	    iteration++;
	    std::cout 
	      << "it: " << iteration << " Cost function L - " 
	      << "L_old: " << L_cmeans_old << " " 
	      << "L_new: " << L_cmeans << " " 
	      << 100 * fabs(L_cmeans-L_cmeans_old) / L_cmeans << "%"
	      << std::endl;

	    // 
	    L_cmeans_old = L_cmeans;
	    L_cmeans = 0;

	    // 
	    // Determins U_ji (Clusters)
	    for( int i = 0 ; i < N ; i++ )
	      if( Bg_mask[i] < 2 )
	      {
		double 
		  x_i         = Points[i],
		  Denominator = 0.;
		// 
		for( int k = 0 ; k < n ; k++ )
		  {
		    double deno  = fabs(x_i - Mu[k]);
		    Denominator += 1./(deno*deno);
		  }
		// 
		for( int j = 0 ; j < n ; j++ )
		  {
		    double numerator = fabs(x_i - Mu[j]);
		    // 
		    numerator *= numerator*Denominator;
		    U_ji[j][i] = 1. / numerator;
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
		  if( Bg_mask[i] < 2 )
		  {
		    // 
		    // 
		    Num += U_ji[j][i]*U_ji[j][i]*static_cast<double>(Points[i]);
		    Den += U_ji[j][i]*U_ji[j][i];
		  }
		//
		Mu[j]  = Num / Den;
	      }

	    // 
	    // Compute the Lagrangien
	    for( int i = 0 ; i < N ; i++ )
	      if( Bg_mask[i] < 2 )
	      {
		double cluster_membership = 0.;
		for( int j = 0 ; j < n ; j++ )
		  {
		    L_cmeans += U_ji[j][i]*U_ji[j][i]*(Points[i]-Mu[j])*(Points[i]-Mu[j]);
		    //
		    if( cluster_membership < U_ji[j][i] ) 
		      {
			cluster_membership = U_ji[j][i];
			Clusters[i] = j;
		      }
		  }
	      }
	      else
		Clusters[i] = 0;
	  }


	//
	//
	std::cout << std::endl;
	std::cout << iteration << std::endl;
	for( int j = 0 ; j < n ; j++ ){
	  std::cout << " " << Mu[j];}
	std::cout << std::endl;


	// 
	// 
	for( int j = 0 ; j < n ; j++ )
	  {
	    delete[] U_ji[j];
	    U_ji[j] = nullptr;
	  }
	// 
	delete[] x;
	x = nullptr;
      }
    }
  }
}
#endif
