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
#ifndef _CUDA_CONDUCTIVITY_MATCHING_FUNCTIONS_
#define _CUDA_CONDUCTIVITY_MATCHING_FUNCTIONS_
#ifdef CUDA
#include <stdio.h>
//
//
//
#define VOXEL_LIMIT 3.65
//
// 600 000      BLOCKS  THREADS REMAIN
//   9 375       9375     64     0
//   4 687.5	 4687    128    64
//   2 343.75	 2343    256   192
//   1 171.875	 1171    512   448  
//
// Threads = 64
//#define THREADS   64
//#define BLOCKS  9375
//#define REMAIN     0
// Threads = 128
#define THREADS  128
#define BLOCKS  4687
#define REMAIN    64
// Threads = 256
//#define THREADS  256
//#define BLOCKS  2343
//#define REMAIN   192
// Threads = 512
//#define THREADS  512
//#define BLOCKS  1171
//#define REMAIN   448
//
//
// 
//
//
//
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file CUDA_Conductivity_matching.h
 * \brief brief explaination 
 * \author Yann Cobigo
 * \version 0.1
 */
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class CUDA_Conductivity_matching_functions
   * \brief classe representing all the GPU device's function
   *
   *  This class representing all the GPU device's function including the kernel.
   */
  __global__ void 
    process_kernel( float* Cell_points, 
		    float* Positions_array_x, 
		    float* Positions_array_y, 
		    float* Positions_array_z,
		    bool*  Do_we_have_conductivity,
		    float* Point_min_distance,
		    int*   Point_min_distance_index)
  {
    //
    //
    int 
      idx = threadIdx.x + blockIdx.x * blockDim.x,
      remain_idx = gridDim.x * blockDim.x + blockIdx.x,
      thread_idx = threadIdx.x,
      block_idx  = blockIdx.x;
    float remaining[REMAIN][5];

    //
    // Shared memory per block
    __shared__ 
      float distance[THREADS][5];
    __shared__ 
      int distance_index[THREADS][5];

    //
    //
    if ( Do_we_have_conductivity[idx] )
      {
	for ( int point = 0 ; point < 5 ; point++ )
	  {
	    distance_index[thread_idx][point] = idx;//threadIdx.x;
	    //    printf("distance_index[%d][%d] = %d\n",thread_idx,i, distance_index[thread_idx][i] );
	    //
	    distance[thread_idx][point] =
	      /*sqrt(*/
	      (Cell_points[3*point+0] - Positions_array_x[idx]) *
	      (Cell_points[3*point+0] - Positions_array_x[idx]) +  
	      (Cell_points[3*point+1] - Positions_array_y[idx]) *
	      (Cell_points[3*point+1] - Positions_array_y[idx]) +  
	      (Cell_points[3*point+2] - Positions_array_z[idx]) *
	      (Cell_points[3*point+2] - Positions_array_z[idx]) /*)*/;
	  }
      }
    else
      {
	for ( int point = 0 ; point < 5 ; point++ )
	  {
	    distance_index[thread_idx][point] = idx;
	    distance[thread_idx][point]       = 1000000.f;
	  }
 
      }
    // Remaining par of output array
    if ( blockIdx.x  < REMAIN && threadIdx.x == 0 )
      {
	if ( Do_we_have_conductivity[remain_idx] )
	  for ( int point = 0 ; point < 5 ; point++ )
	    {
	      remaining[blockIdx.x][point] =
		/*sqrt(*/
		(Cell_points[3*point+0] - Positions_array_x[remain_idx]) *
		(Cell_points[3*point+0] - Positions_array_x[remain_idx]) +  
		(Cell_points[3*point+1] - Positions_array_y[remain_idx]) *
		(Cell_points[3*point+1] - Positions_array_y[remain_idx]) +  
		(Cell_points[3*point+2] - Positions_array_z[remain_idx]) *
		(Cell_points[3*point+2] - Positions_array_z[remain_idx]) /*)*/;
	    }
	else
	  for ( int point = 0 ; point < 5 ; point++ )
	    remaining[blockIdx.x][point] =1000000.f;
      }


    //
    // all threads are done
    __syncthreads();

    //
    // Reductions processus, threadsPerBlock must be a power of 2
    int half = (int)blockDim.x/2;
    //
    while ( half != 0 ) 
      {
	if ( thread_idx < half )
	  for (int point = 0 ; point < 5 ; point++)
	    if ( distance[thread_idx       ][point] > 
		 distance[thread_idx + half][point]  )
	      {
		// copy the smallest distance
		distance[thread_idx][point] = distance[thread_idx + half][point];
		// copy the index
		distance_index[thread_idx][point] = distance_index[thread_idx + half][point];
	      }
	//
	__syncthreads();
	//
	half /= 2;
      }
    //
    if( thread_idx == 0 )
      for (int point = 0 ; point < 5 ; point++)
	{
	  Point_min_distance[5*block_idx + point]       = distance[0][point];
	  Point_min_distance_index[5*block_idx + point] = distance_index[0][point];
	  if ( blockIdx.x  < REMAIN )
	    {
	      Point_min_distance[5*gridDim.x + 5*blockIdx.x + point] = remaining[blockIdx.x][point]; 
	      Point_min_distance_index[5*gridDim.x + 5*blockIdx.x + point] = remain_idx;
	    }
	}
  };



  //__global__ void 
  //process_kernel( float* Cell_points, 
  //		float* Positions_array_x, 
  //		float* Positions_array_y, 
  //		float* Positions_array_z,
  //		float* Point_min_distance,
  //		int*   Point_min_distance_index)
  //{
  ////  printf("Je rentre %d %d %d \n",blockDim.x, THREADS, BLOCKS);
  //  //
  //  //
  //  int 
  //    idx = threadIdx.x + blockIdx.x * blockDim.x,
  //    thread_idx = threadIdx.x,
  //    block_idx  = blockIdx.x;
  //
  //  //
  //  __shared__ 
  //    float distance[THREADS][5];
  //  __shared__ 
  //    int distance_index[THREADS][5];
  //  //
  //  for ( int point = 0 ; point < 5 ; point++ ){
  //    distance_index[thread_idx][point] = idx;//threadIdx.x;
  //    //    printf("distance_index[%d][%d] = %d\n",thread_idx,i, distance_index[thread_idx][i] );
  //    //
  //    distance[thread_idx][point] = (float)
  //      /*sqrt(*/
  //      (Cell_points[3*point+0] - Positions_array_x[idx])*(Cell_points[3*point+0] - Positions_array_x[idx]) +  
  //      (Cell_points[3*point+1] - Positions_array_y[idx])*(Cell_points[3*point+1] - Positions_array_y[idx]) +  
  //      (Cell_points[3*point+2] - Positions_array_z[idx])*(Cell_points[3*point+2] - Positions_array_z[idx]) /*)*/;
  ////    printf("distance[%d][%d] = %f\n",thread_idx,point, distance[thread_idx][point] );
  //
  //  }
  //
  //
  //  //
  //  // all threads are done
  //  __syncthreads();
  //
  //  //
  //  // Reductions processus, threadsPerBlock must be a power of 2
  //  int half = (int)blockDim.x/2;
  //  //
  //  while ( half != 0 ) 
  //    {
  //      if ( thread_idx < half )
  //	for (int point = 0 ; point < 5 ; point++)
  //	  if ( distance[thread_idx       ][point] > 
  //	       distance[thread_idx + half][point] )
  //	    {
  //	      // copy the smallest distance
  //	      distance[thread_idx][point] = distance[thread_idx + half][point];
  //	      // copy the index
  //	      distance_index[thread_idx][point] = distance_index[thread_idx + half][point];
  //	    }
  //      //
  //      __syncthreads();
  //      //
  //      half /= 2;
  //    }
  //  //
  //  if( thread_idx == 0 )
  //    for (int point = 0 ; point < 5 ; point++)
  //      {
  //	Point_min_distance[5*block_idx + point] = distance[0][point];
  //	Point_min_distance_index[5*block_idx + point] = distance_index[0][point];
  //      }
  //};


  //__global__ void 
  //process_kernel( float* Cell_points, 
  //		float* Positions_array_x, 
  //		float* Positions_array_y, 
  //		float* Positions_array_z,
  //		float* Point_min_distance,
  //		int*   Point_min_distance_index)
  //{
  //  //
  //  //
  //  int 
  //    idx = threadIdx.x + blockIdx.x * blockDim.x,
  //    thread_idx = threadIdx.x,
  //    block_idx  = blockIdx.x;
  //  //
  //  __shared__ 
  //    float distance[THREADS][5];
  //  __shared__ 
  //    int distance_index[THREADS][5];
  //
  //  for( int time = 0 ; time < 10 ; time++ )
  //    {
  //      //
  //      //
  //      for ( int point = 0 ; point < 5 ; point++ ){
  //	distance_index[thread_idx][point] = idx;
  //	//
  //	distance[thread_idx][point] = (float)
  //	  /*sqrt(*/
  //	  (Cell_points[15*time+3*point+0] - Positions_array_x[idx])*(Cell_points[15*time+3*point+0] - Positions_array_x[idx]) +  
  //	  (Cell_points[15*time+3*point+1] - Positions_array_y[idx])*(Cell_points[15*time+3*point+1] - Positions_array_y[idx]) +  
  //	  (Cell_points[15*time+3*point+2] - Positions_array_z[idx])*(Cell_points[15*time+3*point+2] - Positions_array_z[idx]) /*)*/;
  //      }
  //
  //
  //      //
  //      // all threads are done
  //      __syncthreads();
  //
  //      //
  //      // Reductions processus, threadsPerBlock must be a power of 2
  //      int half = (int)blockDim.x/2;
  //      //
  //      while ( half != 0 ) 
  //	{
  //	  if ( thread_idx < half )
  //	    for (int point = 0 ; point < 5 ; point++)
  //	      if ( distance[thread_idx       ][point] > 
  //		   distance[thread_idx + half][point] )
  //		{
  //		  // copy the smallest distance
  //		  distance[thread_idx][point] = distance[thread_idx + half][point];
  //		  // copy the index
  //		  distance_index[thread_idx][point] = distance_index[thread_idx + half][point];
  //		}
  //	  //
  //	  __syncthreads();
  //	  //
  //	  half /= 2;
  //	}
  //
  //      //
  //      //
  //      if( thread_idx == 0 )
  //	for (int point = 0 ; point < 5 ; point++)
  //	  {
  //	    Point_min_distance[5*BLOCKS*time + 5*block_idx + point] = distance[0][point];
  //	    Point_min_distance_index[5*BLOCKS*time + 5*block_idx + point] = distance_index[0][point];
  //	  }
  //
  //      //
  //      // all threads are done
  //      __syncthreads();
  //    }
  //};
};
#endif
#endif
