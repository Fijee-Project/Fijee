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
#ifndef DISTANCE_H_
#define DISTANCE_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Distance.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
// UCSF
//
#include "Point_vector.h"
//
// CGAL
//
#include <CGAL/Kd_tree_rectangle.h>
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
  /*! \class Distance
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Distance
  {
  public:
    typedef Point_vector Query_item;
    typedef float FT;

  private:

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Distance
     *
     */
    Distance();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Distance( const Distance& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Distance
     */
    virtual ~Distance();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Distance
     *
     */
    Distance& operator = ( const Distance& );

  public:
    /*!
     */
    float transformed_distance(const Point_vector& P1, const Point_vector& P2) const;
    /*!
     */
    template <class TreeTraits>
      float min_distance_to_rectangle( const Point_vector&,
				       const CGAL::Kd_tree_rectangle<TreeTraits>& ) const;
    /*!
     */
    template <class TreeTraits>
      float max_distance_to_rectangle( const Point_vector&,
				       const CGAL::Kd_tree_rectangle<TreeTraits>& ) const;
    /*!
     */
    float new_distance( float& Dist, float Old_off, float New_off,
			int /* cutting_dimension */) const 
    {
      return Dist + New_off * New_off - Old_off * Old_off;
    }
    /*!
     */
    float transformed_distance(float D) const { return D * D; }
    /*!
     */
    float inverse_of_transformed_distance(float D) { return std::sqrt( D ); }
  };
  //
  //
  //
  template <class TreeTraits>
    float 
    Distance::min_distance_to_rectangle( const Point_vector& p,
					 const CGAL::Kd_tree_rectangle<TreeTraits>& b) const
    {
      float 
	distance(0.0), 
	h = p.x();

      //
      //
      if ( h < b.min_coord(0) ) 
	distance += ( b.min_coord(0) - h ) * ( b.min_coord(0) - h );
      if ( h > b.max_coord(0) ) 
	distance += ( h - b.max_coord(0)) * ( h - b.max_coord(0) );
      //
      h = p.y();
      if ( h < b.min_coord(1) ) 
	distance += ( b.min_coord(1) - h ) * ( b.min_coord(1) - h );
      if ( h > b.max_coord(1) ) 
	distance += ( h - b.max_coord(1) ) * ( h - b.min_coord(1) );
      //
      h = p.z();
      if ( h < b.min_coord(2) ) 
	distance += ( b.min_coord(2) - h ) * ( b.min_coord(2) - h );
      if ( h > b.max_coord(2) ) 
	distance += ( h - b.max_coord(2) ) * ( h - b.max_coord(2) );
      
      //
      //
      return distance;
  }
  //
  //
  //
  template <class TreeTraits>
    float 
    Distance::max_distance_to_rectangle( const Point_vector& p,
					 const CGAL::Kd_tree_rectangle<TreeTraits>& b) const 
    {
      float h = p.x();
      //
      float d0 = ( h >= ( b.min_coord(0) + b.max_coord(0)) / 2.0 ) ?
	( h - b.min_coord(0) ) * ( h - b.min_coord(0) ) : ( b.max_coord(0) - h ) * ( b.max_coord(0) - h );
      //
      h = p.y();
      float d1 = (h >= ( b.min_coord(1) + b.max_coord(1)) / 2.0 ) ?
	( h - b.min_coord(1) ) * ( h - b.min_coord(1) ) : ( b.max_coord(1) - h ) * ( b.max_coord(1) - h );
      //
      h = p.z();
      float d2 = ( h >= ( b.min_coord(2) + b.max_coord(2) ) / 2.0 ) ?
	( h - b.min_coord(2) ) * ( h - b.min_coord(2) ) : ( b.max_coord(2) - h ) * ( b.max_coord(2) - h );

      //
      //
      return d0 + d1 + d2;
  }

  /*!
   *  \brief Dump values for Distance
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Distance : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Distance& );
};
#endif
