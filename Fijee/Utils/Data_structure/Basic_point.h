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
#ifndef BASIC_POINT_H
#define BASIC_POINT_H
/*!
 * \file Basic_point.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Data_structure
  {
    /*! \class Basic_point
     * \brief classe representing a point
     *
     *  This class is a basic data structure for points.
     */
    template< typename Type_point >
      class Basic_point
      {
      private:
	//! weight of the point. This member offert scalar information for the point. 
	Type_point weight_;
	//! position of the point.
	Type_point position_[3];


      public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Basic_point
	 *
	 */
	Basic_point();
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Basic_point
	 *
	 */
	Basic_point( Type_point, Type_point, Type_point, Type_point Weight = 1. );
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Basic_point( const Basic_point& );
	/*!
	 *  \brief Move Constructor
	 *
	 *  Constructor is a moving constructor
	 *
	 */
	Basic_point( Basic_point&& );
	/*!
	 *  \brief Destructeur
	 *
	 *  Destructor of the class Basic_point
	 */
	virtual ~Basic_point(){/*Do nothing*/};
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Basic_point
	 *
	 */
	Basic_point& operator = ( const Basic_point& );
	/*!
	 *  \brief Move Operator =
	 *
	 *  Move operator of the class Basic_point
	 *
	 */
	Basic_point& operator = ( Basic_point&& );
	/*!
	 *  \brief Operator ==
	 *
	 *  Operator compare of the class Basic_point.
	 *
	 */
	bool operator == ( const Basic_point& ) const;
	/*!
	 *  \brief Operator !=
	 *
	 *  Operator different of the class Basic_point.
	 *
	 */
	bool operator != ( const Basic_point& ) const;

      public:
	const Type_point* get_position_() const {return position_; }

	void set_weight_( Type_point Weight ) {weight_ = Weight; }
	Type_point weight() const { return weight_; }
	Type_point x() const { return position_[0]; }
	Type_point y() const { return position_[1]; }
	Type_point z() const { return position_[2]; }

	Type_point& x() { return position_[0]; }
	Type_point& y() { return position_[1]; }
	Type_point& z() { return position_[2]; }

      public:
	/*!
	 *  \brief Squared distance
	 *
	 *  This member compute the squared distance between this point and P.
	 *
	 */
	Type_point squared_distance( const Basic_point& P ) const 
	{ 
	  return 
	    (position_[0]-P.x())*(position_[0]-P.x()) + 
	    (position_[1]-P.y())*(position_[1]-P.y()) + 
	    (position_[2]-P.z())*(position_[2]-P.z()); 
	};
	/*!
	 *  \brief Print XML format
	 *
	 *  This member print out in XML format
	 *
	 */
	std::ostream& print_XML( std::ostream& stream ) const 
	{ 
	  //
	  //
	  stream 
	    << "x=\"" << x() 
	    << "\" y=\"" << y() 
	    << "\" z=\"" << z() << "\" ";
	};
      };
    //
    //
    //
    template< typename Type_point >
      Basic_point<Type_point>::Basic_point(): weight_( static_cast<Type_point>(1.) )
      {
	position_[0] = position_[1] = position_[2] = static_cast<Type_point>(0.);
      }
    //
    //
    //
    template< typename Type_point >
      Basic_point<Type_point>::Basic_point(Type_point X, Type_point Y, Type_point Z, 
					   Type_point Weight ):
    weight_(Weight)
    {
      position_[0] = X;
      position_[1] = Y;
      position_[2] = Z;
    }
    //
    //
    //
    template< typename Type_point >
      Basic_point<Type_point>::Basic_point( const Basic_point<Type_point>& that ): 
    weight_(that.weight_)
      {
	position_[0] = that.position_[0];
	position_[1] = that.position_[1];
	position_[2] = that.position_[2];
      }
    //
    //
    //
    template< typename Type_point >
      Basic_point<Type_point>::Basic_point( Basic_point<Type_point>&& that ): 
    weight_( static_cast<Type_point>(0.) )
      {
	// 
	// 
	position_[0] = static_cast<Type_point>(0.);
	position_[1] = static_cast<Type_point>(0.);
	position_[2] = static_cast<Type_point>(0.);

	// 
	// 
	weight_      = that.weight_;
	position_[0] = that.position_[0];
	position_[1] = that.position_[1];
	position_[2] = that.position_[2];
	
	// 
	// 
	that.weight_      = static_cast<Type_point>(0.);	
	that.position_[0] = static_cast<Type_point>(0.);
	that.position_[1] = static_cast<Type_point>(0.);
	that.position_[2] = static_cast<Type_point>(0.);
      }
    //
    //
    //
    template< typename Type_point > Basic_point<Type_point>& 
      Basic_point<Type_point>::operator = ( const Basic_point<Type_point>& that )
      {
	if( this != &that )
	  {
	    weight_ = that.weight_;
	    //
	    position_[0] = that.position_[0];
	    position_[1] = that.position_[1];
	    position_[2] = that.position_[2];
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > Basic_point<Type_point>& 
      Basic_point<Type_point>::operator = ( Basic_point<Type_point>&& that )
      {
	if( this != &that )
	  {
	    // 
	    // 
	    weight_      = static_cast<Type_point>(0.);
	    position_[0] = static_cast<Type_point>(0.);
	    position_[1] = static_cast<Type_point>(0.);
	    position_[2] = static_cast<Type_point>(0.);
	    
	    // 
	    // 
	    weight_      = that.weight_;
	    position_[0] = that.position_[0];
	    position_[1] = that.position_[1];
	    position_[2] = that.position_[2];
	    
	    // 
	    // 
	    that.weight_      = static_cast<Type_point>(0.);	
	    that.position_[0] = static_cast<Type_point>(0.);
	    that.position_[1] = static_cast<Type_point>(0.);
	    that.position_[2] = static_cast<Type_point>(0.);
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > bool
      Basic_point<Type_point>::operator == ( const Basic_point<Type_point>& that ) const
    {
      return ( weight_      == that.weight_ &&
	       position_[0] == that.position_[0] && 
	       position_[1] == that.position_[1] && 
	       position_[2] == that.position_[2] );
    }
    //
    //
    //
    template< typename Type_point > bool
      Basic_point<Type_point>::operator != ( const Basic_point<Type_point>& that ) const
    {
      return ( weight_      != that.weight_ ||
	       position_[0] != that.position_[0] || 
	       position_[1] != that.position_[1] || 
	       position_[2] != that.position_[2]);
    }
    /*!
     *  \brief Dump values for Basic_point
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     *  \param Basic_point : new position to add in the list
     */
    template< typename Type_point > 
      std::ostream& operator << ( std::ostream& stream, const Basic_point<Type_point>& that )
      {
	//
	//
	stream 
	  << "x= " << that.x() 
	  << " y= " << that.y() 
	  << " z= " << that.z()
	  << " weight= " << that.weight();
	  
	
	//
	//
	return stream;
      };
  }
}
#endif
