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
#ifndef PARCELLATION_METHOD_H
#define PARCELLATION_METHOD_H
/*!
 * \file Parcellation_method.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
//
// UCSF
//
#include "Utils/enum.h"
#include "Parcellation.h"
#include "Parcellation_METIS.h"
#include "Parcellation_Scotch.h"
#include "CGAL_tools.h"
//
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Parcellation_method
   * \brief classe representing whatever
   *
   *  This class uses method library for mesh partioning.
   */
  template < typename Segmentation >
  class Parcellation_method
  {
  private:
    //! Segmentation method
    std::shared_ptr <Segmentation> method_;
    

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Parcellation_method
     *
     */
    Parcellation_method( const C3t3& Mesh, const Cell_pmap& Mesh_map, 
			 const Brain_segmentation Segment, const int N_partitions )
      {method_.reset( new Segmentation(Mesh, Mesh_map, Segment, N_partitions) );};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Parcellation_method( const Parcellation_method& ) = default;
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Parcellation_method
     */
    virtual ~Parcellation_method(){/*Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Parcellation_method
     *
     */
    Parcellation_method& operator = ( const Parcellation_method& ) = default;
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Parcellation_method
     *
     */
    void operator ()()
    { Mesh_partitioning(); };

    
  public:
    /*!
     *  \brief Mesh partitioning
     *
     *  This method uses method library for the partitioning process.
     *
     */
    void Mesh_partitioning()
    {method_->Mesh_partitioning();};
    /*!
      *  \brief Get region
     *
     *  This method return the partition the Elem_number belongs to. 
     *
     *  \param Elem_number
    */
    long int get_region( int Elem_number )
    {return method_->get_region( Elem_number );};
    /*!
     *  \brief Check partitioning
     *
     *  This method true if the number of centroids matching its region is not equal to the number of elements in the sub mesh. 
     *
     *  \param Cells_in_region: number of centroids in the sub mesh.
    */
    bool check_partitioning( int Cells_in_region )
    {return method_->check_partitioning(Cells_in_region);};

  };
};
#endif
