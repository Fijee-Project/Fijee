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
#ifndef ELECTRODE_H
#define ELECTRODE_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Electrode.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
#include <complex>
//
// UCSF
//
#include "Fijee/Fijee_enum.h"
#include "Point.h"
#include "Point_vector.h"
#include "Access_parameters.h"
#include "Electrode_shape.h"
#include "Electrode_shape_sphere.h"
#include "Electrode_shape_cylinder.h"
//
// Eigen
//
#include <Eigen/Dense>
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Electrode
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   * Point_vector discribe the position of the dipole; the direction of the dipole. 
   * member weight_ of Point_vector class represent the intensity (mA) of the dipole.
   */
  class Electrode : public Domains::Point_vector
  {
  private:
    //! Index of the electrode
    int index_;
    //! Label of the electrode
    std::string label_;
    //! Electrode intensity
    float intensity_;
    //! Electrode potential
    float potential_;
    //! Electrode impedance
    std::complex<float> impedance_;
    //! Electrode shape
    std::shared_ptr< Domains::Electrode_shape > shape_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrode
     *
     */
    Electrode();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrode
     *
     */
    Electrode( int, std::string, 
	       float, float, float,
	       float, float, float );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Electrode
     */
    virtual ~Electrode();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Electrode( const Electrode& );
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Electrode
     *
     */
    Electrode& operator = ( const Electrode& );
    
  public:
    ucsf_get_macro(index_, int);
    ucsf_get_macro(label_, std::string);
    ucsf_get_macro(intensity_, float);
    ucsf_get_macro(potential_, float);
    ucsf_get_macro(impedance_, std::complex<float>);
    ucsf_get_macro(shape_, std::shared_ptr< Domains::Electrode_shape > );
    

  public:
    /*!
     *  \brief inside electrode domain
     *
     *  This function check if the point (X, Y, Z) is inside the electrode.
     *
     *  \param X : x-position of the checl point
     *  \param Y : y-position of the checl point
     *  \param Z : z-position of the checl point
     *
     */
    bool inside_domain(float X, float Y, float Z)
    {
      return shape_->inside( *this /*Center*/, X, Y, Z );
    }
 };
  /*!
   *  \brief Dump values for Electrode
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Electrode& );
};
#endif
