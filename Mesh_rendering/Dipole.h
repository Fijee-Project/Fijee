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
#ifndef DIPOLE_H_
#define DIPOLE_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Dipole.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
// UCSF
//
#include "Point_vector.h"
#include "Cell_conductivity.h"
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Dipole
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   * Point_vector discribe the position of the dipole; the direction of the dipole. 
   * member weight_ of Point_vector class represent the intensity (mA) of the dipole.
   */
  class Dipole : public Domains::Point_vector
  {
  private:
    //! Cell number
    int cell_id_;
    //! Cell domain 
    int cell_subdomain_;
    //! Cell parcel 
    int cell_parcel_;
    //! Conductivity coefficients
    //! 0 -> C00
    //! 1 -> C01
    //! 2 -> C02
    //! 3 -> C11
    //! 4 -> C12
    //! 5 -> C22
    float conductivity_coefficients_[6];
    //! Conductivity eigenvalues
    float conductivity_eigenvalues_[3];

    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Dipole
     *
     */
    Dipole();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Dipole( const Dipole& );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor from Cell_conductivity object
     *
     */
    Dipole( const Cell_conductivity& );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor from Cell_conductivity object and a position
     *
     */
    Dipole( const Point_vector& , const Cell_conductivity& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Dipole
     */
    virtual ~Dipole();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Dipole
     *
     */
    Dipole& operator = ( const Dipole& );
    
  public:
    int get_cell_id_() const {return cell_id_;};
    int get_cell_subdomain_() const {return cell_subdomain_;};
    int get_cell_parcel_() const {return cell_parcel_;};

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

    float lambda1() const { return conductivity_eigenvalues_[0]; }
    float lambda2() const { return conductivity_eigenvalues_[1]; }
    float lambda3() const { return conductivity_eigenvalues_[2]; }

    float& lambda1() { return conductivity_eigenvalues_[0]; }
    float& lambda2() { return conductivity_eigenvalues_[1]; }
    float& lambda3() { return conductivity_eigenvalues_[2]; }
 };
  /*!
   *  \brief Dump values for Dipole
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Dipole& );
};
#endif
