#ifndef POINT_H_
#define POINT_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Point.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
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
  /*! \class Point
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Point
  {
  private:
    //! position of the point.
    float position_[3];


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Point
     *
     */
    Point();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Point
     *
     */
    Point( float, float, float );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Point( const Point& );
    //    /*!
    //     *  \brief Move Constructor
    //     *
    //     *  Constructor is a moving constructor
    //     *
    //     */
    //    Point( Point&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Point
     */
    virtual ~Point();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Point
     *
     */
    Point& operator = ( const Point& );
    //    /*!
    //     *  \brief Move Operator =
    //     *
    //     *  Move operator of the class Point
    //     *
    //     */
    //    Point& operator = ( Point&& );
    /*!
     *  \brief Operator ==
     *
     *  Operator compare of the class Point.
     *
     */
    bool operator == ( const Point& );
    /*!
     *  \brief Operator !=
     *
     *  Operator different of the class Point.
     *
     */
    bool operator != ( const Point& );

  public:
    const float* get_position_() const {return position_; }

    float x() const { return position_[0]; }
    float y() const { return position_[1]; }
    float z() const { return position_[2]; }

    float& x() { return position_[0]; }
    float& y() { return position_[1]; }
    float& z() { return position_[2]; }
  };
  /*!
   *  \brief Dump values for Point
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Point& );
};
//
//
//
namespace CGAL {
  template <>
    struct Kernel_traits<Domains::Point> {
    struct Kernel {
      typedef float FT;
      typedef float RT;
    };
  };
}
////
////
////
//struct Construct_coord_iterator {
//  typedef  const float* result_type;
//  const float* operator()(const Domains::Point& p) const
//  { return static_cast<const float*>( p.get_position_() ); }
//
//  const float* operator()(const Domains::Point& p, int)  const
//  { return static_cast<const float*>( p.get_position_() + 3 ); }
//};
#endif
