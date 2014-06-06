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
#ifndef _EMPIRICAL_BAYESIAN_NOISE_H
#define _EMPIRICAL_BAYESIAN_NOISE_H
// 
// Eigen
//
#include <Eigen/Dense>
typedef Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Noise.h"
//
//
//
/*!
 * \file Empirical_bayesian_noise.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Inverse
{
  /*! \class Empirical_bayesian_noise
   * \brief classe representing the source localisation with subtraction method.
   *
   *  This class representing the Physical noise for the source localisation using the subtraction method.
   */
  class Empirical_bayesian_noise : Noise
  {
  private:
    //! Noise covariante matrix
    MatrixXd noise_covariante_matrix_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Empirical_bayesian_noise
     *
     */
  Empirical_bayesian_noise():Noise(){};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Empirical_bayesian_noise( const Empirical_bayesian_noise& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Empirical_bayesian_noise
     */
    virtual ~Empirical_bayesian_noise(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Empirical_bayesian_noise
     *
     */
    Empirical_bayesian_noise& operator = ( const Empirical_bayesian_noise& ){return *this;};

  public:
    /*!
     *  \brief Get noise covariance matrix
     *
     *  This method return the covariance matrix for the noise model.
     *
     */
    virtual inline
      MatrixXd get_covariance() const {return noise_covariante_matrix_;};
  };
}
#endif
