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
#ifndef PARCELLATION_METIS_H
#define PARCELLATION_METIS_H
/*!
 * \file Parcellation_METIS.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <tuple>
#include <algorithm>
#include <string>
//
// UCSF
//
#include "Parcellation.h"
#include "Access_parameters.h"
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
  /*! \class Parcellation_METIS
   * \brief classe representing whatever
   *
   *  This class uses METIS library for mesh partioning.
   */
  class Parcellation_METIS : public Parcellation
  {
  private:
    //! Segment of the brain studied
    Brain_segmentation segment_;
    //! Number of region for the partitioning
    int n_partitions_;
    
    // 
    // Dual graph information
    // 
    
    //! Cells of the mesh in the segmentation
    std::vector< Cell_iterator > elements_nodes_;
    //! Vertex to element structure
    std::map< Tr::Vertex_handle, 
      std::tuple<int/*vertex new id*/, std::list<int/*cell id*/> > > edge_vertex_to_element_;
    // Elements partitioning vector
    std::vector< long int > elements_partitioning_;
    // Nodes partitioning vector
    std::vector< long int > nodes_partitioning_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Parcellation_METIS
     *
     */
    Parcellation_METIS( const C3t3&, const Cell_pmap& , const Brain_segmentation, const int  );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Parcellation_METIS( const Parcellation_METIS& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Parcellation_METIS
     */
    virtual ~Parcellation_METIS();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Parcellation_METIS
     *
     */
    Parcellation_METIS& operator = ( const Parcellation_METIS& );
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Parcellation_METIS
     *
     */
    virtual void operator ()()
    { Mesh_partitioning(); };

    
  public:
    /*!
     *  \brief Mesh partitioning
     *
     *  This method uses METIS library for the partitioning process.
     *
     */
    virtual void Mesh_partitioning( );
    /*!
      *  \brief Get region
     *
     *  This method return the partition the Elem_number belongs to. 
     *
     *  \param Elem_number
    */
    virtual long int get_region( int Elem_number )
    {return elements_partitioning_[Elem_number];};
    /*!
     *  \brief Check partitioning
     *
     *  This method true if the number of centroids matching its region is not equal to the number of elements in the sub mesh. 
     *
     *  \param Cells_in_region: number of centroids in the sub mesh.
    */
    virtual bool check_partitioning( int Cells_in_region )
    {return ( Cells_in_region != static_cast<int>(elements_partitioning_.size()) );};

  private:
    /*!
     */
    virtual void Make_analysis();
  };
  /*!
   *  \brief Dump values for Parcellation_METIS
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Parcellation_METIS: this object
   */
  std::ostream& operator << ( std::ostream&, const Parcellation_METIS& );
};
#endif
