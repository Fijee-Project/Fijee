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
#include "EM_algorithm.h"
// 
// 
// 
//define PI           3.14159265358979323846
// 
// 
typedef Utils::Minimizers::Expectation_maximization UMEM;
// 
// 
// 
UMEM::Expectation_maximization( const int N/*num of sites*/, const unsigned char* Points, const int* Bg_mask,
				int* Clusters,
				const int n/*num of clust*/, double* Mu,
				const Utils::Data_structure::Graph_abstract_data_type<int>& Neighborhood_system ):
  It_minimizer(),
  num_sites_(N), num_sub_sites_(0), measure_(Points), bg_mask_(Bg_mask), clusters_(Clusters), 
  num_clusters_(n), mu_(Mu)
{
  // 
  // 
  sigma_2_ = new double[n];
  nk_      = new int[n];
  //
  for( int j = 0 ; j < n ; j++ )
    {
      sigma_2_[j] = 0.;
      nk_[j]      = 0;
    }
 
  // 
  // 
  neighborhood_system_ = Neighborhood_system;

  // 
  // Buid the variance vector
  if( bg_mask_ )
    {
      for( int i = 0 ; i < N ; i++ )
	if( bg_mask_[i] < 2 )
	  {
	    // 
	    // 
	    num_sub_sites_++;
	    //
	    for( int j = 0 ; j < n ; j++ )
	      sigma_2_[j] += (mu_[j] - measure_[i])*(mu_[j] - measure_[i]);
	    // 
	    nk_[clusters_[i]] += 1;
	  }
      //
      for( int j = 0 ; j < n ; j++ )
	sigma_2_[j] /= static_cast<double>(num_sub_sites_);
    }
  else
    {
      for( int i = 0 ; i < N ; i++ )
	{
	  for( int j = 0 ; j < n ; j++ )
	    sigma_2_[j] += (mu_[j] - measure_[i])*(mu_[j] - measure_[i]);
	  // 
	  nk_[clusters_[i]] += 1;
	}
      //
      for( int j = 0 ; j < n ; j++ )
	sigma_2_[j] /= static_cast<double>(N);
    }
}
// 
// 
// 
void
UMEM::minimize()
{
  if(true)
    GMM_EM2();
  else
    HMRF_EM();
}
// 
// 
// 
void 
UMEM::HMRF_EM()
{}
// 
// 
// 
void 
UMEM::GMM_EM()
{
  // 
  // 
  std::cout << "EM algorithm on Random Markov Field" << std::endl;

  // 
  // 
  double mu_old[num_clusters_];
  double convergence = 0.;
  int iteration = 0;

  // 
  // Alpha_k
  std::vector< double > a_k ( num_clusters_, 0. );
  for( int j = 0 ; j < num_clusters_ ; j ++ )
    a_k[j] = static_cast<double>(nk_[j]) / static_cast<double>(num_sites_);

  //
  // 
  while( iteration++ < 20 )
    {
      // 
      // 
      std::cout << "EM iteration: " << iteration << std::endl;

      // 
      // Build p_ik
      // p_ik = P(l|X_{neighbor})*sqrt(1/(2*Pi*sigma_sqared_k))*exp{-(y_{i} - mu_k)²/(2*sigma_sqared_k)}
      std::vector< std::vector<double> > p_ik( num_sites_, std::vector<double>( num_clusters_, 1. ) );
      std::vector< double >              p_i ( num_sites_, 0. );
      std::vector< std::vector<double> > q_ik( num_sites_, std::vector<double>( num_clusters_, 1. ) );
      std::vector< double >              q_k ( num_clusters_, 0. );
      // 
      for(int i = 0 ; i < num_sites_ ; i++ )
	for(int j = 0 ; j < num_clusters_ ; j++ )
	  {
	    // 
	    // 
	    double C = sqrt(2*M_PI*sigma_2_[j]);
	    if( C != 0. )
	      C = 1./C;
	    else
	      {
		std::cerr << "C should not be 0 !" << std::endl;
		abort();
	      }
	    // Normal part of the probability distribution
	    double 
	      arg = (static_cast<double>(measure_[i]) - mu_[j])*(static_cast<double>(measure_[i]) - mu_[j]);
	    arg  /= -(2*sigma_2_[j]);
	    // 
	    p_ik[i][j] *= C * std::exp( arg );
	    p_ik[i][j] *= a_k[j];
	    // 
//	    if(p_ik[i][j] < 0. || p_ik[i][j] > 1.)
//	      {
//		std::cout << "p_ik["<<i<<"]["<<j<<"] " << p_ik[i][j]
//			  << "\nC " << C 
//			  << "\n measure_["<<i<<"] " << static_cast<int>(measure_[i])
//			  << "\n mu_["<<j<<"] " << mu_[j]
//			  << "\n sigma_2_["<<j<<"] " << sigma_2_[j]
//			  << "\n a_k["<<j<<"] " << a_k[j]
//			  << std::endl;
//		exit(-1);
//	      }
	    // Integration over clusters for the site i
	    p_i[i] += p_ik[i][j];
	  }

      // 
      // Build q_ik at the step n
      for(int j = 0 ; j < num_clusters_ ; j++ )
	for(int i = 0 ; i < num_sites_ ; i++ )
	  {
	    q_ik[i][j] = p_ik[i][j] / p_i[i];
	    q_k[j]    += q_ik[i][j];
	  }

      // 
      // New clustering
      for(int i = 0 ; i < num_sites_ ; i++ )
	{
	  int    cluster = 0;
	  double proba_k = 0.;
	  // 
	  for( int j = 0 ; j < num_clusters_ ; j++ )
	    if( proba_k < q_ik[i][j] )
	      {
		proba_k = q_ik[i][j];
		cluster = j;
	      }
	  // 
	  clusters_[i] = cluster;
	}

      // 
      // Sufficient statistics
      // 

      // 
      // Build mu_k and sigma_squared_k at the step (n+1)
      for(int j = 0 ; j < num_clusters_ ; j++ )
	{
	  // 
	  // Reset variables
	  mu_old[j] = mu_[j];
	  // 
	  double new_mu = 0.;
	  double new_sigma2 = 0.;
	  // 
	  for(int i = 0 ; i < num_sites_ ; i++ )
	    new_mu     += q_ik[i][j] * measure_[i];
	  // 
	  mu_[j] = new_mu / q_k[j];
	  a_k[j] = q_k[j] / static_cast<double>(num_sites_);
	  // 
	  for(int i = 0 ; i < num_sites_ ; i++ )
	    new_sigma2 += q_ik[i][j] * (mu_[j] - measure_[i])*(mu_[j] - measure_[i]);
	  // 
	  sigma_2_[j] = new_sigma2 / q_k[j];
	}

      // 
      //
      convergence = 0.;
      for( int j = 0 ; j < num_clusters_ ; j++ )
	{
	  convergence += (mu_[j]-mu_old[j])*(mu_[j]-mu_old[j]);
	  std::cout << " " << mu_[j];
	}
      std::cout << std::endl;
      std::cout << convergence << std::endl;
    }
}
// 
// 
// 
void 
UMEM::GMM_EM2()
{
  // 
  // 
  std::cout << "EM algorithm on Random Markov Field" << std::endl;

  // 
  // 
  double mu_old[num_clusters_];
  double convergence = 0.;
  int iteration = 0;

  // 
  // Alpha_k
  std::vector< double > a_k ( num_clusters_, 0. );
  for( int j = 0 ; j < num_clusters_ ; j ++ )
    a_k[j] = static_cast<double>(nk_[j]) / static_cast<double>(num_sub_sites_);

  //
  // 
  while( iteration++ < 50 )
    {
      // 
      // 
      std::cout << "EM iteration: " << iteration << std::endl;

      // 
      // Build p_ik
      // p_ik = P(l|X_{neighbor})*sqrt(1/(2*Pi*sigma_sqared_k))*exp{-(y_{i} - mu_k)²/(2*sigma_sqared_k)}
      std::vector< std::vector<double> > p_ik( num_sites_, std::vector<double>( num_clusters_, 1. ) );
      std::vector< double >              p_i ( num_sites_, 0. );
      std::vector< std::vector<double> > q_ik( num_sites_, std::vector<double>( num_clusters_, 1. ) );
      std::vector< double >              q_k ( num_clusters_, 0. );
      // 
      for(int i = 0 ; i < num_sites_ ; i++ )
	if( bg_mask_[i] < 2 )
	  for(int j = 0 ; j < num_clusters_ ; j++ )
	    {
	      // 
	      // 
	      double C = sqrt(2*M_PI*sigma_2_[j]);
	      if( C != 0. )
		C = 1./C;
	      else
		{
		  std::cerr << "C should not be 0 !" << std::endl;
		  abort();
		}
	      // Normal part of the probability distribution
	      double 
		arg = (static_cast<double>(measure_[i]) - mu_[j])*(static_cast<double>(measure_[i]) - mu_[j]);
	      arg  /= -(2*sigma_2_[j]);
	      // 
	      p_ik[i][j] *= C * std::exp( arg );
	      p_ik[i][j] *= a_k[j];
	      // 
//	    if(p_ik[i][j] < 0. || p_ik[i][j] > 1.)
//	      {
//		std::cout << "p_ik["<<i<<"]["<<j<<"] " << p_ik[i][j]
//			  << "\nC " << C 
//			  << "\n measure_["<<i<<"] " << static_cast<int>(measure_[i])
//			  << "\n mu_["<<j<<"] " << mu_[j]
//			  << "\n sigma_2_["<<j<<"] " << sigma_2_[j]
//			  << "\n a_k["<<j<<"] " << a_k[j]
//			  << std::endl;
//		exit(-1);
//	      }
	      // Integration over clusters for the site i
	      p_i[i] += p_ik[i][j];
	    }


      // 
      // Build q_ik at the step n
      for( int j = 0 ; j < num_clusters_ ; j++ )
	for( int i = 0 ; i < num_sites_ ; i++ )
	  if( bg_mask_[i] < 2 )
	    {
	      q_ik[i][j] = p_ik[i][j] / p_i[i];
	      q_k[j]    += q_ik[i][j];
	      //	      std::cout << "q_ik["<<i<<"]["<<j<<"]" << q_ik[i][j] << std::endl;
	    }
//	  else
//	    {
//	      q_ik[i][j] = ( j == 0 ? 1. : 0. );
//	      //	      q_k[j] = 1.;
//	    }

      // 
      // New clustering
      for(int i = 0 ; i < num_sites_ ; i++ )
	if( bg_mask_[i] < 2 )
	{
	  int    cluster = 0;
	  double proba_k = 0.;
	  // 
	  for( int j = 0 ; j < num_clusters_ ; j++ )
	    if( proba_k < q_ik[i][j] )
	      {
		proba_k = q_ik[i][j];
		cluster = j;
	      }
	  // 
	  clusters_[i] = cluster;
	}
	else
	  clusters_[i] = 0;


      // 
      // Sufficient statistics
      // 

      // 
      // Build mu_k and sigma_squared_k at the step (n+1)
      for(int j = 0 ; j < num_clusters_ ; j++ )
	{
	  // 
	  // Reset variables
	  mu_old[j] = mu_[j];
	  // 
	  double new_mu = 0.;
	  double new_sigma2 = 0.;
	  // 
	  for(int i = 0 ; i < num_sites_ ; i++ )
	    if( bg_mask_[i] < 2 )
	      new_mu     += q_ik[i][j] * measure_[i];
	  // 
	  mu_[j] = new_mu / q_k[j];
	  a_k[j] = q_k[j] / static_cast<double>(num_sub_sites_);
	  // 
	  for(int i = 0 ; i < num_sites_ ; i++ )
	    if( bg_mask_[i] < 2 )
	      new_sigma2 += q_ik[i][j] * (mu_[j] - measure_[i])*(mu_[j] - measure_[i]);
	  // 
	  sigma_2_[j] = new_sigma2 / q_k[j];
	}

      // 
      //
      convergence = 0.;
      for( int j = 0 ; j < num_clusters_ ; j++ )
	{
	  convergence += (mu_[j]-mu_old[j])*(mu_[j]-mu_old[j]);
	  std::cout << " " << mu_[j];
	}
      std::cout << std::endl;
      std::cout << convergence << std::endl;
    }
}
