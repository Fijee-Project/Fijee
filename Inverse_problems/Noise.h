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
#ifndef _NOISE_H
#define _NOISE_H
#include <fstream>  
// 
// Eigen
//
#include <Eigen/Dense>
typedef Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;
//
// UCSF project
//
#include "Utils/Fijee_environment.h"

//using namespace dolfin;

/*! \namespace Inverse
 * 
 * Name space for our new package
 *
 */
//
// 
//
namespace Inverse
{
  /*! \class Physics
   * \brief classe representing the mother class of all physical type of noises
   *
   *  This class representing the 
   */
  class Noise
  {
  public:
//    /*!
//     *  \brief Default Constructor
//     *
//     *  Constructor of the class Noise
//     *
//     */
//    Noise(){};
//    /*!
//     *  \brief Copy Constructor
//     *
//     *  Constructor is a copy constructor
//     *
//     */
//    Noise( const Noise& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Noise
     */
    virtual ~Noise(){/* Do nothing */};
//    /*!
//     *  \brief Operator =
//     *
//     *  Operator = of the class Noise
//     *
//     */
//    Noise& operator = ( const Noise& ){return *this;};
//    /*!
//     *  \brief Operator ()
//     *
//     *  Operator () of the class Noise
//     *
//     */
//    virtual void operator ()() = 0;
    /*!
     *  \brief Get noise covariance matrix
     *
     *  This method return the covariance matrix for the noise model.
     *
     */
    virtual inline
     MatrixXd get_covariance() const = 0;
   };
};

#endif
