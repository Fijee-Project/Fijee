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
#ifndef PARCELLATION_SCOTCH_H
#define PARCELLATION_SCOTCH_H
/*!
 * \file Parcellation_Scotch.h
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
  /*! \class Parcellation_Scotch
   * \brief classe representing whatever
   *
   *  This class uses Scotch library for mesh partioning.
   */
  class Parcellation_Scotch : public Parcellation
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
     *  Constructor of the class Parcellation_Scotch
     *
     */
    Parcellation_Scotch( const C3t3&, const Cell_pmap&, const Brain_segmentation, const int  );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Parcellation_Scotch( const Parcellation_Scotch& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Parcellation_Scotch
     */
    virtual ~Parcellation_Scotch();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Parcellation_Scotch
     *
     */
    Parcellation_Scotch& operator = ( const Parcellation_Scotch& );
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Parcellation_Scotch
     *
     */
    virtual void operator ()()
    { Mesh_partitioning(); };

  public:
    /*!
     *  \brief Mesh partitioning
     *
     *  This method uses Scotch library for the partitioning process.
     *
     *  \param 
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

  private:
    /*!
     */
    virtual void Make_analysis();
  };
  /*!
   *  \brief Dump values for Parcellation_Scotch
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Parcellation_Scotch: this object
   */
  std::ostream& operator << ( std::ostream&, const Parcellation_Scotch& );
};
#endif
