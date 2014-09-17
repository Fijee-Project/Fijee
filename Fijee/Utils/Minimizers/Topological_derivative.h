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
#ifndef TOPOLOGICAL_DERIVATIVE_H
#define TOPOLOGICAL_DERIVATIVE_H
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
    namespace Topological_derivative
    {
      /*
       * \brief 
       *
       * This fonction compute 
       * Note: the algorithm can be optimized and transfer on GPU. 
       *
       */
      template < typename T, typename U > void Discrete_topological_derivative( const int N, const T* Points, int* Clusters,
										const int n, double* Mu, double* Variance,
										int* Element_per_cluster,
										double Theta = 0.5,
										double Intensity_min = 0.,
										double Intensity_max = 255.,
										double Epsilon = 1.e-01 )
      {
	// 
	// Time log
	FIJEE_TIME_PROFILER("Utils::Minimizers::Topological_derivative::Lloyd_algorythm");

	// 
	// 
	double delta = 1. / Intensity_max; // Boundaries sensitivity
	double set_size = Intensity_max - Intensity_min;
	double domaine  = set_size / (double) n;
	double min_max_cluster[n][2];
	// positions Mu in the center of domaines
	for( int j = 0 ; j < n ; j++ )
	  {
	    Mu[j] = domaine/2. + j*domaine;
	    min_max_cluster[j][0] = j*domaine;
	    min_max_cluster[j][1] = (j+1)*domaine;
	  }

	// 
	// Initialization
	double theta = 1.;
	double** derive_topo          = (double**) malloc( N * sizeof(double) );
	double*  normalized_intensity = (double*)  malloc( N * sizeof(double) );
	double*  normalized_Mu        = (double*)  malloc( n * sizeof(double) );
	// 
	for ( int i = 0 ; i < N ; i++ )
	  {
	    normalized_intensity[i] = static_cast<double>(Points[i]) / Intensity_max;
	    derive_topo[i] = (double*) malloc( n * sizeof(double) );
	    Clusters[i] = 0;
	    
	  }
	// 
	for ( int j = 0 ; j < n ; j++ )
	  {
	    normalized_Mu[j] = static_cast<double>(Mu[j]) / Intensity_max;
	    min_max_cluster[j][0] /= Intensity_max; // min
	    min_max_cluster[j][1] /= Intensity_max; // max
	  }
	  
	// 
	// Computes the topological derivative
	// 

	// 
	// First step theta = 1
	// 

	// 
	// 
	bool do_it_again = true;
	bool first_step = true;
	double
	  Phi = 0.,
	  Chi = 0.;
	double D_min    = 1.e+09;
	int D_min_index = -1;


	// 
	// 
	while ( do_it_again )
	  {
	    // 
	    // 
	    do_it_again = false;
	    
	    // 
	    // Check if we are in the first step
	    if ( first_step )
	      {
		theta = 1.;
		first_step = false;
	      }
	    else
	      theta = Theta;
		
	    // 
	    // 
	    for ( int i = 0 ; i < N ; i++ )
	      {
		// 
		D_min = 1.e+09;
		D_min_index = -1;
		//
		for ( int j = 0 ; j < n ; j++ )
		  {
		    // 
		    // 
		    Phi  = (normalized_intensity[i] - normalized_Mu[j]) * (normalized_intensity[i] - normalized_Mu[j]);
		    Phi -= (normalized_intensity[i] - normalized_Mu[Clusters[i]])*(normalized_intensity[i] - normalized_Mu[Clusters[i]]);
		    // 
		    if ( Clusters[i] != j )
		      Chi = ( fabs(normalized_intensity[i] - min_max_cluster[j][0]) < 1.5*delta ||  fabs(normalized_intensity[i] - min_max_cluster[j][1]) < 1.5*delta ? 
			      1 : 0 )/(4.*1. /*dimension*/); // we just have the color as dimension.
		    // 
		    derive_topo[i][j] = theta * Phi + (1 - theta) * Chi;
		    //		  std::cout << "(i,j) " << i << ", " << j << std::endl;
		    //		  std::cout << "normalized_intensity[i]: " << normalized_intensity[i] << std::endl;
		    //		  std::cout << "normalized_Mu[j]: " << normalized_Mu[j] << std::endl;
		    //		  std::cout << "Points[i]: " << static_cast<double>(Points[i]) << std::endl;
		    //		  std::cout << "Mu[j]: " << Mu[j] << std::endl;
		    //		  std::cout << "Clusters[i]: " << Clusters[i] << std::endl;
		    //		  std::cout << "normalized_Mu[Clusters[i]]: " << normalized_Mu[Clusters[i]] << std::endl;
		  
		    // 
		    // 
		    if( derive_topo[i][j] < D_min )
		      {
			D_min = derive_topo[i][j];
			D_min_index = j;
		      }
		    //		  std::cout << "D_min: " << D_min << std::endl;
		    //		  std::cout << "D_min_index: " << D_min_index << std::endl;
		    //		  std::cout << "derive_topo: " << derive_topo[i][j] << std::endl;

		  }

		// 
		// 
		if ( D_min < 0 )
		  {
		    Clusters[i] = D_min_index;
		    do_it_again = true;
		  }
	      }

	    // 
	    // Recompute Mu, min and max of the cluster
	    double 
	      Num = 0., 
	      Den = 0.;
	    for( int j = 0 ; j < n ; j++ )
	      {
		// 
		// 
		Num = 0.;
		Element_per_cluster[j] = 0.;
		// 
		min_max_cluster[j][0] = Intensity_max;
		min_max_cluster[j][1] = Intensity_min;
		// 
		//	    
		for( int i = 0 ; i < N ; i++ )
		  if( Clusters[i] == j )
		    {
		      Num += static_cast<double>(Points[i]);
		      Element_per_cluster[j] += 1.;
		      // 
		      if ( static_cast<double>(Points[i]) < min_max_cluster[j][0] )
			min_max_cluster[j][0] = static_cast<double>(Points[i]);
		      if ( static_cast<double>(Points[i]) > min_max_cluster[j][1] )
			min_max_cluster[j][1] = static_cast<double>(Points[i]);
		    }
		//
		if( Element_per_cluster[j] != 0 )
		  Mu[j] = static_cast<double>(Num) / static_cast<double>(Element_per_cluster[j]);
		
		normalized_Mu[j] = Mu[j] / Intensity_max; 
		min_max_cluster[j][0] /= Intensity_max; 
		min_max_cluster[j][1] /= Intensity_max; 
		std::cout <<  Mu[j] << " " << normalized_Mu[j]
			  << " min: " << min_max_cluster[j][0] << " max: " << min_max_cluster[j][1] 
			  << std::endl;;
	      }
	  }

	// 
	// 
	delete[] normalized_intensity;
	normalized_intensity = nullptr;
	// 
	//	  for ( int i = 0 ; i < N ; i++ )
	//	    {
	//	      delete[] normalized_intensity[i];
	//	      normalized_intensity[i] = nullptr;
	//	    }
	delete[] normalized_intensity;
	normalized_intensity = nullptr;
      }


      /*
       * \brief 
       *
       * This fonction compute 
       *
       *
       */
      template < typename T, typename U > void Discrete_topological_derivative( const int N, 
										const T* Points_T1, int* Clusters_T1,
										const T* Points_T2, int* Clusters_T2,
										const int n, 
										double* Mu_T1, double* Variance_T1,
										double* Mu_T2, double* Variance_T2,
										int* Element_per_cluster_T1,
										int* Element_per_cluster_T2,
										double Intensity_min_T1 = 0., double Intensity_max_T1 = 255.,
										double Intensity_min_T2 = 0., double Intensity_max_T2 = 255.,
										double Theta = 0.5 )
      {
	// 
	// Time log
	FIJEE_TIME_PROFILER("Utils::Minimizers::Topological_derivative::Lloyd_algorythm");

	// 
	// 
	int iteration = 0;
	//
	double delta = 1. / 255.; // Boundaries sensitivity
	double set_size_T1 = Intensity_max_T1 - Intensity_min_T1;
	double set_size_T2 = Intensity_max_T2 - Intensity_min_T2;
	double domain_T1  = set_size_T1 / (double) n;
	double domain_T2  = set_size_T2 / (double) n;
	double min_max_cluster_T1[n][2];
	double min_max_cluster_T2[n][2];
	int order_T1[n];
	int order_T2[n];
	// positions Mu_T1 in the center of domain_T1s
	for( int j = 0 ; j < n ; j++ )
	  {
	    Mu_T1[j] = domain_T1/2. + j*domain_T1;
	    min_max_cluster_T1[j][0] = j*domain_T1;
	    min_max_cluster_T1[j][1] = (j+1)*domain_T1;
	    order_T1[j] = order_T2[j] = 0;
	    // 
	    Mu_T2[j] = domain_T2/2. + j*domain_T2;
	    min_max_cluster_T2[j][0] = j*domain_T2;
	    min_max_cluster_T2[j][1] = (j+1)*domain_T2;
	  }

	// 
	// Initialization
	double theta = 1.;
	double* derive_topo              = (double*) malloc( N * sizeof(double) );
	//
	double*  normalized_intensity_T1 = (double*)  malloc( N * sizeof(double) );
	double*  normalized_Mu_T1        = (double*)  malloc( n * sizeof(double) );
	// 
	double*  normalized_intensity_T2 = (double*)  malloc( N * sizeof(double) );
	double*  normalized_Mu_T2        = (double*)  malloc( n * sizeof(double) );
	// 
	for ( int i = 0 ; i < N ; i++ )
	  {
	    normalized_intensity_T1[i] = static_cast<double>(Points_T1[i]) / Intensity_max_T1;
	    Clusters_T1[i] = 0;
	    //
	    normalized_intensity_T2[i] = static_cast<double>(Points_T2[i]) / Intensity_max_T2;
	    Clusters_T2[i] = 0;
	  }
	// 
	for ( int j = 0 ; j < n ; j++ )
	  {
	    normalized_Mu_T1[j] = static_cast<double>(Mu_T1[j]) / Intensity_max_T1;
	    min_max_cluster_T1[j][0] /= Intensity_max_T1; // min
	    min_max_cluster_T1[j][1] /= Intensity_max_T1; // max
	    //
	    normalized_Mu_T2[j] = static_cast<double>(Mu_T2[j]) / Intensity_max_T2;
	    min_max_cluster_T2[j][0] /= Intensity_max_T2; // min
	    min_max_cluster_T2[j][1] /= Intensity_max_T2; // max
	  }
	  
	// 
	// Computes the topological derivative
	// 

	// 
	// First step theta = 1
	// 

	// 
	// 
	bool do_it_again = true;
	bool first_step = true;
	double
	  Phi = 0.,
	  Chi = 0.;
	double D_min    = 1.e+09;
	int D_min_index[2] = {-1, -1};

	// 
	// 
	while ( do_it_again )
	  {
	    // 
	    // 
	    iteration++;
	    do_it_again = false;
	    std::cout << "iteration: " << iteration << std::endl;
	    
	    // 
	    // Check if we are in the first step
	    if ( first_step )
	      {
		theta = 1.;
		first_step = false;
	      }
	    else
	      theta = Theta;
		
	    // 
	    // 
	    for ( int i = 0 ; i < N ; i++ )
	      {
		// 
		D_min = 1.e+09;
		D_min_index[0] = -1;
		D_min_index[1] = -1;
		//
		for ( int j = 0 ; j < n ; j++ )
		  for ( int k = 0 ; k < n ; k++ )
		    {
		      // 
		      // 
		      Phi  = (normalized_intensity_T1[i] - normalized_Mu_T1[j]) * (normalized_intensity_T1[i] - normalized_Mu_T1[j]);
		      Phi += (normalized_intensity_T2[i] - normalized_Mu_T2[k]) * (normalized_intensity_T2[i] - normalized_Mu_T2[k]);
		      Phi -= (normalized_intensity_T1[i] - normalized_Mu_T1[Clusters_T1[i]])*(normalized_intensity_T1[i] - normalized_Mu_T1[Clusters_T1[i]]);
		      Phi -= (normalized_intensity_T2[i] - normalized_Mu_T2[Clusters_T2[i]])*(normalized_intensity_T2[i] - normalized_Mu_T2[Clusters_T2[i]]);
		      // 
		      if ( Clusters_T1[i] != j )
			Chi = ( fabs(normalized_intensity_T1[i] - min_max_cluster_T1[j][0]) < 1.5*delta ||  fabs(normalized_intensity_T1[i] - min_max_cluster_T1[j][1]) < 1.5*delta ? 
				1 : 0 )/(4.*1. /*dimension*/); // we just have the color as dimension.
		      if ( Clusters_T2[i] != j )
			Chi += ( fabs(normalized_intensity_T2[i] - min_max_cluster_T2[k][0]) < 1.5*delta ||  fabs(normalized_intensity_T2[i] - min_max_cluster_T2[k][1]) < 1.5*delta ? 
				 1 : 0 )/(4.*1. /*dimension*/); // we just have the color as dimension.
		      // 
		      derive_topo[i] = theta * Phi + (1 - theta) * Chi;
		      
		      // 
		      // 
		      if( derive_topo[i] < D_min )
			{
			  D_min = derive_topo[i];
			  D_min_index[0] = j;
			  D_min_index[1] = k;
			}
		    }

		// 
		// 
		if ( D_min < 0 )
		  {
 		    Clusters_T1[i] = D_min_index[0];
		    Clusters_T2[i] = D_min_index[1];
		    //		    do_it_again = true;
		  }
	      }

	    // 
	    // Recompute Mu_T1, min and max of the cluster
	    double 
	      Num_T1 = 0., 
	      Num_T2 = 0.,
	      Den_T1 = 0., 
	      Den_T2 = 0.;
	    //
	    for( int j = 0 ; j < n ; j++ )
	      {
		// 
		// 
		Num_T1 = 0.;
		Num_T2 = 0.;
		Den_T1 = 0.;
		Den_T2 = 0.;
		// 
		min_max_cluster_T1[j][0] = Intensity_max_T1;
		min_max_cluster_T1[j][1] = Intensity_min_T1;
		// 
		min_max_cluster_T2[j][0] = Intensity_max_T2;
		min_max_cluster_T2[j][1] = Intensity_min_T2;
		//	    
		for( int i = 0 ; i < N ; i++ )
		  {
		    if( Clusters_T1[i] == j )
		      {
			Num_T1 += static_cast<double>(Points_T1[i]);
			Den_T1 += 1.;
			// 
			if ( static_cast<double>(Points_T1[i]) < min_max_cluster_T1[j][0] )
			  min_max_cluster_T1[j][0] = static_cast<double>(Points_T1[i]);
			if ( static_cast<double>(Points_T1[i]) > min_max_cluster_T1[j][1] )
			  min_max_cluster_T1[j][1] = static_cast<double>(Points_T1[i]);
		      }
		    //
		    if( Clusters_T2[i] == j )
		      {
			Num_T2 += static_cast<double>(Points_T2[i]);
			Den_T2 += 1.;
			// 
			if ( static_cast<double>(Points_T2[i]) < min_max_cluster_T2[j][0] )
			  min_max_cluster_T2[j][0] = static_cast<double>(Points_T2[i]);
			if ( static_cast<double>(Points_T2[i]) > min_max_cluster_T2[j][1] )
			  min_max_cluster_T2[j][1] = static_cast<double>(Points_T2[i]);
		      }
		  }
		//
		if( Den_T1 != 0 )
		  {
		    Mu_T1[j] = static_cast<double>(Num_T1) / Den_T1;
		    order_T1[j] = Den_T1;
		    if( Den_T1 != Element_per_cluster_T1[j] )
		      {
			do_it_again = true;
			Element_per_cluster_T1[j] = Den_T1;
		      }
		  }
		//
		if( Den_T2 != 0 )
		  {
		    Mu_T2[j] = static_cast<double>(Num_T2) / Den_T2;
		    order_T2[j] = Den_T2;
		    if( Den_T2 != Element_per_cluster_T2[j] )
		      {
			do_it_again = true;
			order_T2[j] = Element_per_cluster_T2[j] = Den_T2;
		      }
		  }
		// 		
		normalized_Mu_T1[j] = Mu_T1[j] / Intensity_max_T1; 
		min_max_cluster_T1[j][0] /= Intensity_max_T1; 
		min_max_cluster_T1[j][1] /= Intensity_max_T1; 
		// 
		normalized_Mu_T2[j] = Mu_T2[j] / Intensity_max_T2; 
		min_max_cluster_T2[j][0] /= Intensity_max_T2; 
		min_max_cluster_T2[j][1] /= Intensity_max_T2; 


		std::cout 
		  <<  Mu_T1[j] << " " << normalized_Mu_T1[j]
		  << " min: " << min_max_cluster_T1[j][0] << " max: " << min_max_cluster_T1[j][1] 
		  << " num_of_elem: " << Element_per_cluster_T1[j]
		  << "\t"
		  <<  Mu_T2[j] << " " << normalized_Mu_T2[j]
		  << " min: " << min_max_cluster_T2[j][0] << " max: " << min_max_cluster_T2[j][1] 
		  << " num_of_elem: " << Element_per_cluster_T2[j]
		  << std::endl;;
	      }
	    // 
	    // 
	    // order the clusters
	    int permutation = 1;
	    double tempo = 0.;
	    //
	    while( permutation != 0 )
	      {
		permutation = 0; 
		// 
		for ( int j = 0 ; j < n-1 ; j++ )
		  {
		    if( order_T1[j] > order_T1[j+1] )
		      {
			tempo = order_T1[j];
			order_T1[j] = order_T1[j+1];
			order_T1[j+1] = tempo;
			permutation++;
		      }
		    // 
		    if( order_T2[j] > order_T2[j+1] )
		      {
			tempo = order_T2[j];
			order_T2[j] = order_T2[j+1];
			order_T2[j+1] = tempo;
			permutation++;
		      }
		  }
	      }
	    std::cout <<  std::endl;
	    for( int j = 0 ; j < n ; j++ )
	      std::cout << "order_T1[" << j << "] = " << order_T1[j] << " ";
	    std::cout <<  std::endl;
	    for( int j = 0 ; j < n ; j++ )
	      std::cout << "order_T2[" << j << "] = " << order_T2[j] << " ";
	    std::cout <<  std::endl;
	  }

	// 
	// 
	delete[] normalized_intensity_T1;
	normalized_intensity_T1 = nullptr;
	delete[] normalized_intensity_T2;
	normalized_intensity_T2 = nullptr;
	// 
	//	  for ( int i = 0 ; i < N ; i++ )
	//	    {
	//	      delete[] normalized_intensity_T1[i];
	//	      normalized_intensity_T1[i] = nullptr;
	//	    }
	delete[] normalized_intensity_T1;
	normalized_intensity_T1 = nullptr;
	delete[] normalized_intensity_T2;
	normalized_intensity_T2 = nullptr;
      }
    }
  }
}
#endif
