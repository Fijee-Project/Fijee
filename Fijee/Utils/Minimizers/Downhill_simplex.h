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
#ifndef DOWNHILL_SIMPLEX_H
#define DOWNHILL_SIMPLEX_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Downhill_simplex_sphere.h
 * \brief brief describe 
 * \author John Zheng He, Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <vector>
#include <algorithm>    // std::sort
//
// UCSF
//
#include "Minimizer.h"
//
//
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
    /*! \class Shape
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class Downhill_simplex : public It_minimizer
    {
    private:
      //! Simplex vertices
      std::vector< Estimation_tuple > simplex_;
      //! Map of conductivity boundary values for each tissues
      std::vector< std::tuple<double, double> > conductivity_boundaries_;
      //! Funtion to minimize
      Function function_;
      //! Tolerance
      double delta_;
      //! Reflection coefficient
      double reflection_coeff_;
      //! Expension coefficient
      double expension_coeff_;
      //! Contraction coefficient
      double contraction_coeff_;
      //! Dimension of the space: (N_ + 1) number of vertices in a simplex
      double N_;

    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Downhill_simplex_sphere
       *
       */
      Downhill_simplex();
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Downhill_simplex_sphere
       */
      virtual ~Downhill_simplex(){/* Do nothing*/};
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
    Downhill_simplex( const Downhill_simplex& that):It_minimizer(that){};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Downhill_simplex_sphere
       *
       */
      Downhill_simplex& operator = ( const Downhill_simplex& that)
	{
	  It_minimizer::operator=(that);
	  return *this;
	};

      
    public:
      /*!
       *  \brief minimize function
       *
       *  This method initialize the minimizer
       */
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& );
      /*!
       *  \brief minimize function
       *
       *  This method launch the minimization algorithm
       */
      virtual void minimize();

    private:
      /*!
       *  \brief Order the simplex vertices
       *
       *  This method order the simplex vertices
       */
      void order_vertices();
      /*!
       *  \brief get facet centroid
       *
       *  This method 
       */
      const Eigen::Vector3d get_facet_centroid() const;
      /*!
       *  \brief Convergence criteria
       *
       *  This method 
       */
      bool is_converged();
      /*!
       *  \brief Contraction
       *
       *  This method TODO
       */
      void contraction();
      /*!
       *  \brief Transform
       *
       *  This method TODO
       */
      void transform(){};
      /*!
       *  \brief Reflection
       *
       *  This method TODO
       */
      Eigen::Vector3d reflection();
      /*!
       *  \brief Get middle
       *
       *  This method TODO
       */
      const Eigen::Vector3d get_middle( const Eigen::Vector3d&, 
					const Eigen::Vector3d& ) const;
    };
    /*!
     *  \brief Dump values for Electrode
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     */
    std::ostream& operator << ( std::ostream&, const Downhill_simplex& );
  }
}
#endif
