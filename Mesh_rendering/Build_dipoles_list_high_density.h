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
#ifndef BUILD_DIPOLES_LIST_HIGH_DENSITY_H
#define BUILD_DIPOLES_LIST_HIGH_DENSITY_H
/*!
 * \file Build_dipoles_list_high_density.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <list>
#include <tuple>
//
// UCSF
//
#include "Build_dipoles_list.h"
#include "Access_parameters.h"
#include "Point_vector.h"
#include "Dipole.h"
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/basic.h>
// dD Spatial Searching
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
// Point Set Processing
#include <CGAL/compute_average_spacing.h>
#include <CGAL/grid_simplify_point_set.h>
// Point Set Processing
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Dipole_position;
typedef boost::tuple<int, Dipole_position, Domains::Point_vector> IndexedPointVector;
// dD Spatial Searching
typedef CGAL::Search_traits_3< Kernel >Traits_base;
typedef std::tuple< Dipole_position , Domains::Cell_conductivity > High_density_key_type;
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
  // -----------------------------------
  // K-nearest neighbor algorithm (CGAL)
  // -----------------------------------
  struct Point_vector_high_density_map
  {
    typedef Dipole_position value_type;
    typedef const value_type& reference;
    typedef const High_density_key_type& key_type;
    typedef boost::readable_property_map_tag category;
  };
  // get function for the property map
  Point_vector_high_density_map::reference 
    get( Point_vector_high_density_map, Point_vector_high_density_map::key_type p);
  //
  typedef CGAL::Search_traits_adapter< High_density_key_type, Point_vector_high_density_map, Traits_base > High_density_traits;
  typedef CGAL::Orthogonal_k_neighbor_search< High_density_traits > High_density_neighbor_search;
  typedef High_density_neighbor_search::Tree High_density_tree;
  typedef High_density_neighbor_search::Distance High_density_distance;

  // -----------------------------------

  /*! \class Build_dipoles_list_high_density
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Build_dipoles_list_high_density : public Build_dipoles_list
  {
  private:
    //! Left hemisphere white matter point-vectors
    std::list< Point_vector > lh_wm_;
    //! Right hemisphere white matter point-vectors
    std::list< Point_vector > rh_wm_;
    //! Dipoles list
    std::list< Domains::Dipole > dipoles_list_;
    //! Cell size controlling the inter dipoles distance. This variable allow to control the density of the dipole distribution
    double cell_size_;
    //! Gray matter layer populated by the dipole distribution. If layer_ = 1, only the first layer of gray matter centroids will be populated. This variable allow to control the density of the dipole distribution
    int layer_;
#ifdef TRACE
#if TRACE == 100
    //! Mesh centroid and white matter point-vector tuples list
    std::list< std::tuple< Point_vector, Point_vector > > centroid_vertex_;
#endif
#endif      

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Build_dipoles_list_high_density
     *
     */
    Build_dipoles_list_high_density();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Build_dipoles_list_high_density( const Build_dipoles_list_high_density& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Build_dipoles_list_high_density
     */
    virtual ~Build_dipoles_list_high_density();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Build_dipoles_list_high_density
     *
     */
    Build_dipoles_list_high_density& operator = ( const Build_dipoles_list_high_density& );
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Build_dipoles_list_high_density
     *
     */
    virtual void operator ()()
    {
      Output_dipoles_list_xml();
    };

  public:
    /*!
     */
    virtual void Make_list( const std::list< Cell_conductivity >& List_cell_conductivity );
    /*!
     */
    virtual void Build_stream(std::ofstream& );
    /*!
     *  \brief Output the XML of the dipoles' list
     *
     *  This method create the list of dipoles.
     *
     */
    virtual void Output_dipoles_list_xml();

  private:
    /*!
     */
    virtual void Make_analysis();
    /*!
     */
    void Select_dipole( const High_density_tree&, 
			const std::vector< IndexedPointVector >&, 
			std::vector< bool >&  );
  };
  /*!
   *  \brief Dump values for Build_dipoles_list_high_density
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Build_dipoles_list_high_density: this object
   */
  std::ostream& operator << ( std::ostream&, const Build_dipoles_list_high_density& );
};
#endif
