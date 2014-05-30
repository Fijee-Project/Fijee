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
#ifndef ELECTRODE_SHAPE_H
#define ELECTRODE_SHAPE_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Electrode.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
//
// UCSF
//
#include "Point.h"
#include "Point_vector.h"
#include "Access_parameters.h"
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

  /*! \class Electrode shape
   * \brief classe representing whatever
   *
   *  This class is an example of class 
   * 
   */
  class Electrode_shape
  {
  public:
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Electrode_shape_sphere
     */
    virtual ~Electrode_shape(){/* Do nothing */};  

  public:
    /*!
     *  \brief inside 
     *
     *  This function check if the point (X, Y, Z) is inside the shape.
     *
     */
    virtual bool inside( Domains::Point_vector&, 
			 float, float, float ) const = 0;
    /*!
     *  \brief Size of the contact surface
     *
     *  This function return the size of the contect surface between the electrode and the scalp.
     *
     */
    virtual float contact_surface() const = 0;
    /*!
     *  \brief print members
     *
     *  This function print the class members
     *
     */
    virtual void print( std::ostream& ) const = 0;
  };
};
#endif
