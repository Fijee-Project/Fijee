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
    //! weight of the point. This member offert scalar information for the point. 
    float weight_;


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
    Point( float, float, float, float Weight = 1. );
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

    void set_weight_( float Weight ) {weight_ = Weight; }
    float weight() const { return weight_; }
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
#endif
