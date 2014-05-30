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
#ifndef MINIMIZER_H
#define MINIMIZER_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Minimization.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <tuple>
#include <functional>
//
// Eigen
//
#include <Eigen/Dense>
//
// UCSF
//
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \namespace Minimizers
   * 
   * Name space for our new package
   *
   */
  namespace Minimizers
  {
    typedef std::function< double( const Eigen::Vector3d& ) > Function;
    typedef std::tuple< 
      double,          /* - 0 - estimation */
      Eigen::Vector3d /* - 1 - sigma (0) skin, (1) skull compacta, (2) skull spongiosa */
      > Estimation_tuple;
    /*! \class Minimizer
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class Minimizer
    {
    public:
      virtual ~Minimizer(){/* Do nothing */};  
      
    public:
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& ) = 0;
      virtual void minimize() = 0;
    };
    /*! \class It_minimizer
     * \brief classe representing the 
     *
     *  This class is an example of class I will have to use
     */
    class It_minimizer : public Minimizer
    {
    protected:
      //! Number of iteration
      int iteration_;
      //! Max number of iterations
      int max_iterations_;

    public:
      /*!
       *  \brief Constructor
       *
       *  Constructor of the class Minimizer
       */
    It_minimizer():iteration_(0), max_iterations_(200){};
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Minimizer
       */
      virtual ~It_minimizer(){/* Do nothing */};
            /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
    It_minimizer( const It_minimizer& that):
      iteration_(that.iteration_), max_iterations_(that.max_iterations_){};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class It_minimizer_sphere
       *
       */
      It_minimizer& operator = ( const It_minimizer& that )
	{
	  iteration_      = that.iteration_;
	  max_iterations_ = that.max_iterations_;
	  //
	  return *this;
	};

    public:
      /*!
       *  \brief initialization function
       *
       *  This method initialize the minimizer
       */
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& ) = 0;
      /*!
       *  \brief minimize function
       *
       *  This method launch the minimization algorithm
       */
      virtual void minimize() = 0;
    };
  }
}
#endif
