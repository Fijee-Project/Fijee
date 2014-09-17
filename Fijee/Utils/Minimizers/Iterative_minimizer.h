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
#ifndef ITERATIVE_MINIMIZER_H
#define ITERATIVE_MINIMIZER_H
#include <list>
#include <memory>
#include <string>
#include <functional>
//
// UCSF project
//
/*!
 * \file Iterative_minimizer.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Minimizers
  {
    typedef std::function< double( const Eigen::Vector3d& ) > Function;
    typedef std::tuple< 
      double,          /* - 0 - estimation */
      Eigen::Vector3d /* - 1 - sigma (0) skin, (1) skull spongiosa, (2) skull compacta */
      > Estimation_tuple;

   /*! \class Iterative_minimizer
     * \brief classe representing the dipoles distribution
     *
     *  This class is an example of class I will have to use
     */
    template < typename Minimizer_algo >
      class Iterative_minimizer
      {
      private:
	Minimizer_algo minimizer_;

      public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Iterative_minimizer
	 *
	 */
	Iterative_minimizer(){};
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Iterative_minimizer( const Iterative_minimizer& ){};
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Iterative_minimizer
	 *
	 */
	Iterative_minimizer& operator = ( const Iterative_minimizer& ){};
	//    /*!
	//     *  \brief Operator ()
	//     *
	//     *  Operator () of the class Iterative_minimizer
	//     *
	//     */
	//    void operator () ();

      public:
	/*!
	 *  \brief initialization function
	 *
	 *  This method initialized the minimizer
	 */
	void initialization( Function Fun,  
			     const std::vector< Estimation_tuple >& Simplex,
			     const std::vector< std::tuple<double, double> >& Boundaries)
	{
	  minimizer_.initialization( Fun, Simplex, Boundaries );
	};
	/*!
	 *  \brief minimize function
	 *
	 *  This method launch the minimization algorithm
	 */
	void minimize()
	{
	  minimizer_.minimize();
	};
      };
  }
}
#endif
