#ifndef POINT_VECTOR_H_
#define POINT_VECTOR_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Point_vector.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
//
//
#include "Point.h"
//
// CGAL
//
#include <CGAL/Simple_cartesian.h>
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Point_vector
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Point_vector : public Domains::Point
  {
  private:
  //! Vector at the Point.
  float vector_[3];
  //! Vector at the Point.
  float norm_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Point_vector
     *
     */
    Point_vector();
     /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Point_vector
     *
     */
    Point_vector( float, float, float,
		  float, float, float, float Weight = 1. );
   /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Point_vector( const Point_vector& );
//    /*!
//     *  \brief Move Constructor
//     *
//     *  Constructor is a moving constructor
//     *
//     */
//    Point_vector( Point_vector&& ) = delete;
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Point_vector
     */
    virtual ~Point_vector();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Point_vector
     *
     */
    Point_vector& operator = ( const Point_vector& );
//    /*!
//     *  \brief Move Operator =
//     *
//     *  Move operator of the class Point_vector
//     *
//     */
//    Point_vector& operator = ( Point_vector&& ) = delete;
    /*!
     *  \brief Operator ==
     *
     *  Operator compare of the class Point_vector.
     *
     */
    bool operator == ( const Point_vector& );
    /*!
     *  \brief Operator !=
     *
     *  Operator different of the class Point_vector.
     *
     */
    bool operator != ( const Point_vector& );
    /*!
     *  \brief Operator +
     *
     *  Operator addition of the class Point_vector.
     *
     */
     Point_vector& operator + ( const Point_vector& );
    /*!
     *  \brief Operator +=
     *
     *  Operator addition of the class Point_vector.
     *
     */
     Point_vector& operator += ( const Point_vector& );
    /*!
     *  \brief Operator -
     *
     *  Operator addition of the class Point_vector.
     *
     */
     Point_vector& operator - ( const Point_vector& );
    /*!
     *  \brief Operator -=
     *
     *  Operator addition of the class Point_vector.
     *
     */
     Point_vector& operator -= ( const Point_vector& );

  public:
    /*!
     *  \brief get norm
     *
     *  This method is the vector's norm accessor.
     *
     */
    float get_norm_() const {return norm_; }
    /*!
     *  \brief get vector
     *
     *  This method is the vector accessor.
     *
     */
    const float* get_vector_() const {return vector_; }

    float vx() const { return vector_[0]; };
    float vy() const { return vector_[1]; };
    float vz() const { return vector_[2]; };

    float& vx() { return vector_[0]; };
    float& vy() { return vector_[1]; };
    float& vz() { return vector_[2]; };
    /*!
     *  \brief cross
     *
     *  This method is the cross product for the Point_vector's vector.
     *
     *   \param Vector: for the cross product.
     */
    Point_vector&  cross( const Point_vector& Vector );
    /*!
     *  \brief dot
     *
     *  This method is the inner product for the Point_vector's vector.
     *
     *   \param Vector: for the inner product.
     */
    float dot( const Point_vector& Vector ) const;
    /*!
     *  \brief cosine_theta
     *
     *  This method is the normalized inner product for the Point_vector's vector.
     *
     *   \param Vector: for the inner product.
     */
    float cosine_theta( const Point_vector& Vector ) const;

  private:
    /*!
     *  \brief normalize
     *
     *  This method is renormalized the Point_vector's vector.
     *
     */
    void normalize();
  };
  /*!
   *  \brief Dump values for Point_vector
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Point_vector& );
};
//
//
//
namespace CGAL {
  template <>
    struct Kernel_traits<Domains::Point_vector> {
    struct Kernel {
      typedef float FT;
      typedef float RT;
    };
  };
}
//
//
//
struct Construct_coord_iterator {
  typedef  const float* result_type;
  const float* operator()(const Domains::Point_vector& p) const
  { return static_cast<const float*>( p.get_position_() ); }

  const float* operator()(const Domains::Point_vector& p, int)  const
  { return static_cast<const float*>( p.get_position_() + 3 ); }
};
#endif
