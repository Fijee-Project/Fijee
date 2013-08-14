#ifndef CUDA_CONDUCTIVITY_MATCHING_H_
#define CUDA_CONDUCTIVITY_MATCHING_H_
#include <iostream>
//
// UCSF
//
//#include "enum.h"
//
// CUDA runtime
//
//#include <cuda_runtime.h>
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
  /*! \class CUDA_Conductivity_matching
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class CUDA_Conductivity_matching
  {
  private:
    //! Size of array
    int size_of_array_;
    //! __device__ positions X of voxel centroid array
    float *positions_array_x_;
    //! __device__ positions Y of voxel centroid array
    float *positions_array_y_;
    //! __device__ positions Z of voxel centroid array
    float *positions_array_z_;

  public:
    CUDA_Conductivity_matching();
    CUDA_Conductivity_matching(int, 
			       float*,
			       float*,
			       float*
			       );
//    CUDA_Conductivity_matching( const Domain& );
//    CUDA_Conductivity_matching( Domain&& );
    ~CUDA_Conductivity_matching();
    //
//    operator = ( const Domain& );
//    operator = ( Domain&& );
    
  public:
    /*!
     *  \brief find the vertices conductivity index
     *
     *  This method return 
     *
     *  \param Vertices_position: is a table of 5 x 3 floating points representing the position of vertices and centroid.
     *
     *  \return vertices_voxel_index: is a 5 integers array with conductivity index for each vertices. i = 0,1,2,3 for the vertices and i = 4 for the centroid.
     */
    int* find_vertices_voxel_index(float* Vertices_position);

// private:
//   /*!
//    *  \brief Kernel
//    *
//    *  This method return the 
//    *
//    *  \param 
//    *
//    */
//   __global__ void process_kernel(){};
  };
  /*!
   *  \brief Dump values for CUDA_Conductivity_matching
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const CUDA_Conductivity_matching& );
};
#endif
