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
#ifndef _SPHERES_ELECTRIC_MONOPOLE_H
#define _SPHERES_ELECTRIC_MONOPOLE_H
#include <cmath>
#include <math.h>
#include <algorithm>    // std::copy
//
// FEniCS
//
#include <dolfin.h>
//
// Boost
//
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
//
// Eigen
//
#include <Eigen/Dense>
//
//
//
#define NUM_SPHERES 4
#define NUM_ITERATIONS 300
//
//
//
using namespace dolfin;
/*!
 * \file Source.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Solver
{
  /*! \class Spheres_electric_monopole
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Spheres_electric_monopole : public Expression
  {
  private:
    //! Position of the injection
    Point r0_values_;
    //! Intensity of the injection [I_] = A
    double I_;

    //
    // Concentric spheres
    // 
    //! Spheres radius [mm]
    //! r_{1} scalp
    //! r_{2} skull
    //! r_{3} CSF
    //! r_{4} brain
    double r_sphere_[NUM_SPHERES];
    //! Spheres conductivity [S/m]
    //! \varepsilon_{i,j}
    //! i = 0, 1, 2, 3 speres radius 1, 2, 3 and 4
    //! j = 1, 2 conductivity longitudinal and orthogonal
    double sigma_[NUM_SPHERES][2];
    //! Development exposant
    double nu_[NUM_ITERATIONS][NUM_SPHERES];
    //! Coefficients A_{j}^{(1,2)} and B_{j}^{(1,2)}
    Eigen::Matrix <double, 2, 1>  A_B_[NUM_ITERATIONS][NUM_SPHERES][2];
    //! Transfere matrix
    Eigen::Matrix <double, 2, 2> M_[NUM_ITERATIONS][NUM_SPHERES];
    //! Transfere matrix inv
    Eigen::Matrix <double, 2, 2> M_Inv[NUM_ITERATIONS][NUM_SPHERES];
    //! Transfere matrix deter
    double M_det[NUM_ITERATIONS][NUM_SPHERES];
    //! Radial function coefficient
    double R_coeff_[NUM_ITERATIONS];


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Spheres_electric_monopole
     *
     */
    Spheres_electric_monopole();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Spheres_electric_monopole
     *
     */
    Spheres_electric_monopole(const double, const Point&);
    /*!
     *  \brief Copy Constructor
     *
     *  Copy constructor of the class Spheres_electric_monopole
     *
     */
    Spheres_electric_monopole( const Spheres_electric_monopole& );
    /*!
     *  \brief destructor
     *
     *  Destructo of the class Spheres_electric_monopole
     *
     */
    ~Spheres_electric_monopole(){/* Do nothing */};

  public:
    /*!
     *  \brief Operator =
     *
     *  Copy constructor of the class Spheres_electric_monopole
     *
     */
    Spheres_electric_monopole& operator =( const Spheres_electric_monopole& );

  private:
    /*!
     */
    virtual void eval(Array<double>& values, const Array<double>& x, const ufc::cell&) const;
    /*!
     *  \brief Radial part
     *
     *  This method process the radial part of the solution at \vec{r} = \vec{x} and the rank {\it n}
     * 
     */
    double R( const int, const double ) const;
    /*!
     *  \brief Radial sub part
     *
     *  This method process the radial sub part.
     * 
     */
    double R( const int, const int, const int, const double ) const;
    /*!
     *
     *  \brief Angular part
     *
     *  This method process the angular part of the solution  \vec{r} = \vec{x} and the rank {\it n}. The solution has an azimuthal symetry, the the function are Legendre polynomial of the first kind.
     */
    double P( const int, const Point& )const;
    /*!
     *
     *  \brief Spherical harmonics
     *
     *  This method process the angular part of the solution  \vec{r} = \vec{x} and the rank {\it n}.
     */
    double Yn( const int, const Point& )const;
  };
  /*!
   *  \brief Dump values for Spheres_electric_monopole
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Spheres_electric_monopole& );

}
#endif
