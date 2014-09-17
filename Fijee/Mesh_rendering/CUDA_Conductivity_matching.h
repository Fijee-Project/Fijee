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
#ifndef CUDA_CONDUCTIVITY_MATCHING_H_
#define CUDA_CONDUCTIVITY_MATCHING_H_
#include <iostream>
#include <stdio.h>
//
// UCSF
//
//#include "Utils/enum.h"
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
    //! __device__ conductivity in the voxel
    bool  *do_we_have_conductivity_;

  public:
    CUDA_Conductivity_matching();
    CUDA_Conductivity_matching(int, 
			       float*,
			       float*,
			       float*,
			       bool*
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
     *  \return Vertices_voxel: is a 5 x BLOCKS floating points array with the smallest distance per blocks for each vertices. i = 0,1,2,3 for the vertices and i = 4 for the centroid.
     *
     *  \return Vertices_voxel_index: is a 5 x BLOCKS integers array with conductivity index corresponding to the distances  in Vertices_voxel array for each vertices. i = 0,1,2,3 for the vertices and i = 4 for the centroid.
     */
    void find_vertices_voxel_index( float* Vertices_position, 
				    float* Vertices_voxel, 
				    int*   Vertices_voxel_index);
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
