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
#ifndef LABELED_DOMAIN_H_
#define LABELED_DOMAIN_H_
#include <iostream>
#include <string>
#include <fstream>
//
// UCSF
//
#include "Fijee/Fijee_enum.h"
//
// Project
//
//
//
//
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Labeled_domain.h
 * \brief brief explaination 
 * \author Yann Cobigo
 * \version 0.1
 */
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Labeled_domain
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  // Implicite_domain = 
  template< typename Implicite_domain, typename Point_type, typename VectorPointNormal>
  class Labeled_domain
  {
  private:
    Implicite_domain* implicite_domain_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Labeled_domain
     *
     */
  Labeled_domain():
    implicite_domain_( nullptr )
      {};
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Labeled_domain
     *
     */
    Labeled_domain( const char* File ):
    implicite_domain_( new Implicite_domain( File ) )
      {};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Labeled_domain( const Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >& )
      {};
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Labeled_domain( Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >&& )
      {};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Labeled_domain
     */
    virtual ~Labeled_domain()
      {
	delete implicite_domain_;
	implicite_domain_ = NULL;
      };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Labeled_domain
     *
     */
    Labeled_domain& operator = ( const Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >& )
      {};
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Labeled_domain
     *
     */
    Labeled_domain& operator = ( Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >&& )
      {};
    /*!
     *  \brief Move Operator ()
     *
     *  Move operator of the class Labeled_domain
     *
     */
    void operator ()( double** Positions )
    {
      (*implicite_domain_)( Positions );
    };

  public:
    /*!
     *  \brief Get max_x_ value
     *
     *  This method return the maximum x coordinate max_x_.
     *
     *  \return max_x_
     */
    //    inline double get_max_x( ) const {return max_x_;};
 
  public:
    /*!
     *  \brief Inside domain
     *
     *  This method check if a point is inside the implicite domain
     *
     */
    inline bool inside_domain( Point_type Point_Type )
    {
      return implicite_domain_->inside_domain( Point_Type );
    };
    /*!
     *  \brief
     */
    inline const double* get_poly_data_bounds_()
    {
      return implicite_domain_->get_poly_data_bounds_();
    };
    /*!
     *  \brief Get point_normal vector
     *
     *  This method return point_normal_ of the STL mesh.
     *
     *  \return point_normal_
     */
    inline VectorPointNormal get_point_normal()
    {
      return implicite_domain_->get_point_normal_();
    };
  };
  /*!
   *  \brief Dump values for Labeled_domain
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
//  std::ostream& operator << ( std::ostream&, const Labeled_domain< Implicite_domain >& );
};
#endif
