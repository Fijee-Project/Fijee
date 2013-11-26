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
