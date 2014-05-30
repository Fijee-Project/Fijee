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
#ifndef CELL_CONDUCTIVITY_H_
#define CELL_CONDUCTIVITY_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Cell_conductivity.h
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
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Cell_conductivity
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Cell_conductivity
  {
  private:
    //! Cell number
    int cell_id_;
    //! Cell domain 
    int cell_subdomain_;
    //! Conductivity coefficients
    //! 0 -> C00
    //! 1 -> C01
    //! 2 -> C02
    //! 3 -> C11
    //! 4 -> C12
    //! 5 -> C22
    float conductivity_coefficients_[6];
    //! Point vector centroid with eigenvectors | l1 >, | l2 > et | l3 >
    Domains::Point_vector centroid_lambda_[3];

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Cell_conductivity
     *
     */
    Cell_conductivity();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Cell_conductivity
     *
     */
    Cell_conductivity( int, int, 
		       float, float, float,
		       float, float, float, float, 
		       float, float, float, float, 
		       float, float, float, float,
		       float, float, float, float, float, float );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Cell_conductivity( const Cell_conductivity& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Cell_conductivity
     */
    virtual ~Cell_conductivity();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Cell_conductivity
     *
     */
    Cell_conductivity& operator = ( const Cell_conductivity& );

  public:
    int get_cell_id_() const {return cell_id_;};
    int get_cell_subdomain_() const {return cell_subdomain_;};

    const float* get_conductivity_coefficients_() const {return conductivity_coefficients_; }

    float C00() const { return conductivity_coefficients_[0]; }
    float C01() const { return conductivity_coefficients_[1]; }
    float C02() const { return conductivity_coefficients_[2]; }
    float C11() const { return conductivity_coefficients_[3]; }
    float C12() const { return conductivity_coefficients_[4]; }
    float C22() const { return conductivity_coefficients_[5]; }

    float& C00() { return conductivity_coefficients_[0]; }
    float& C01() { return conductivity_coefficients_[1]; }
    float& C02() { return conductivity_coefficients_[2]; }
    float& C11() { return conductivity_coefficients_[3]; }
    float& C12() { return conductivity_coefficients_[4]; }
    float& C22() { return conductivity_coefficients_[5]; }

    const Domains::Point_vector* get_centroid_lambda_() const {return centroid_lambda_; }

  };
  /*!
   *  \brief Dump values for Cell_conductivity
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Cell_conductivity& );
};
#endif
