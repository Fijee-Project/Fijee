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
#ifndef BUILD_DIPOLES_LIST_KNN_H_
#define BUILD_DIPOLES_LIST_KNN_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Build_dipoles_list_knn.h
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
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Orthogonal_incremental_neighbor_search.h>
//
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Dipole_position;
typedef CGAL::Search_traits_3< Kernel >Traits_base;
//
typedef std::tuple< Dipole_position, std::tuple<Domains::Point_vector, Domains::Point_vector> > Key_type;
//typedef std::tuple< Dipole_position, int > Key_type;
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
  struct Point_vector_property_map
  {
    typedef Dipole_position value_type;
    typedef const value_type& reference;
    typedef const Key_type& key_type;
    typedef boost::readable_property_map_tag category;
  };
  // get function for the property map
  Point_vector_property_map::reference 
    get( Point_vector_property_map, Point_vector_property_map::key_type p);
  //
  typedef CGAL::Search_traits_adapter< Key_type, Point_vector_property_map, Traits_base > Dipoles_traits;
  typedef CGAL::Orthogonal_k_neighbor_search< Dipoles_traits > Dipoles_neighbor_search;
  typedef Dipoles_neighbor_search::Tree Dipoles_tree;
  typedef Dipoles_neighbor_search::Distance Dipole_distance;
  //
  // Check the distance between two dipoles
  //
  typedef CGAL::Orthogonal_incremental_neighbor_search< Traits_base > incremental_search;
  typedef incremental_search::iterator knn_iterator;
  typedef incremental_search::Tree Distance_tree;

  // -----------------------------------

  /*! \class Build_dipoles_list_knn
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Build_dipoles_list_knn : public Build_dipoles_list
  {
  private:
    //! Couple of the closest white and gray matter vertices in the left hemisphere 
    std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > > lh_match_wm_gm_;
    //! Couple of the closest white and gray matter vertices in the right hemisphere 
    std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > > rh_match_wm_gm_;
    //! Dipoles list
    std::list< Domains::Dipole > dipoles_list_;
#ifdef TRACE
#if TRACE == 100
    //! 
    std::list< std::tuple< Domains::Point_vector, std::tuple< Domains::Point_vector, Domains::Point_vector > > > match_centroid_wm_gm_;
#endif
#endif      

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Build_dipoles_list_knn
     *
     */
    Build_dipoles_list_knn();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Build_dipoles_list_knn( const Build_dipoles_list_knn& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Build_dipoles_list_knn
     */
    virtual ~Build_dipoles_list_knn();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Build_dipoles_list_knn
     *
     */
    Build_dipoles_list_knn& operator = ( const Build_dipoles_list_knn& );
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Build_dipoles_list_knn
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
    bool Select_dipole( Dipoles_tree&, Domains::Point_vector& );
  };
  /*!
   *  \brief Dump values for Build_dipoles_list_knn
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Build_dipoles_list_knn: this object
   */
  std::ostream& operator << ( std::ostream&, const Build_dipoles_list_knn& );
};
#endif
