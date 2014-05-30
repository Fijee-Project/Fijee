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
#ifndef _CONDUCTIVITY_H
#define _CONDUCTIVITY_H
#include <dolfin.h>
#include <vector>
//
// Eigen
//
#include <Eigen/Dense>
//
// UCSF
//
#include "PDE_solver_parameters.h"
//
//
//
using namespace dolfin;
//
/*!
 * \file Conductivity.h
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
  /*! \class Tensor_conductivity
   * \brief classe representing anisotrope conductivity
   *
   *  This class is an example of class I will have to use
   */
  class Tensor_conductivity : public Expression
  {
  private:
    // tensor elements
    //! C00 conductivity tensor elements
    MeshFunction<double> C00_;
    //! C01 conductivity tensor elements
    MeshFunction<double> C01_;
    //! C02 conductivity tensor elements
    MeshFunction<double> C02_;
    //! C11 conductivity tensor elements
    MeshFunction<double> C11_;
    //! C12 conductivity tensor elements
    MeshFunction<double> C12_;
    //! C22 conductivity tensor elements
    MeshFunction<double> C22_;

    //! Skull spongiosa isotropic conductivity
    double sigma_skull_spongiosa_;
    //! Skull compacta isotropic conductivity
    double sigma_skull_compacta_;
    //! Skin isotropic conductivity
    double sigma_skin_;


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Tensor_conductivity
     *
     */
    Tensor_conductivity( std::shared_ptr< const Mesh > );
    /*!
     *  \brief destructor
     *
     *  Constructor of the class Tensor_conductivity
     *
     */
    ~Tensor_conductivity(){/*Do nothing*/};

    
  public:
    /*!
     *  \brief eval
     *
     *  This method evaluates expression on each cell
     *
     */
    virtual void eval(Array<double>& , const Array<double>& , const ufc::cell& ) const;
    /*!
     *  \brief value_rank
     *
     *  This method returns the rank of the tensor
     *
     */
    virtual std::size_t value_rank() const
      {
	return 2;
      }
    /*!
     *  \brief value_dimension
     *
     *  This method evaluates 
     *
     */
    virtual std::size_t value_dimension(uint i) const
      {
	return 3;
      }

  public:
     /*!
     *  \brief conductivity_update
     *
     *  This method update the conductivity in the skin/skull electrical conductivity estimation 
     *
     */
    void conductivity_update( const std::shared_ptr< MeshFunction< std::size_t > >,
			      const Eigen::Vector3d& );
  };
  
  /*! \class Sigma_isotrope
   * \brief classe representing isotrope conductivity
   *
   *  This class is an example of class I will have to use
   */
  class Sigma_isotrope : public Expression
  {
  private:
    //! isotrope value of the conductivity tensor
    double sigma_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Sigma_isotrope
     *
     */
  Sigma_isotrope( double Sigma = 1) : Expression(3,3), sigma_(Sigma) {}

  public:
    /*!
     *  \brief eval
     *
     *  This method evaluates expression on each cell
     *
     */
    void eval(Array<double>& , const Array<double>& ) const;
    /*!
     *  \brief value_rank
     *
     *  This method evaluates 
     *
     */
    virtual std::size_t value_rank() const
      {
	return 2;
      }
    /*!
     *  \brief value_dimension
     *
     *  This method evaluates 
     *
     */
    virtual std::size_t value_dimension(uint i) const
      {
	return 3;
      }
  };

  /*! \class Sigma_skull
   * \brief classe representing anisotrope of the skull
   *
   *  This class is an example of class I will have to use
   */
  class Sigma_skull : public Expression
  {
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Tensor_conductivity
     *
     */
  Sigma_skull() : Expression(2,2) {}

  public:
    /*!
     *  \brief eval
     *
     *  This method evaluates expression on each cell
     *
     */
    void eval(Array<double>& , const Array<double>& ) const;
    /*!
     *  \brief value_rank
     *
     *  This method evaluates 
     *
     */
    virtual std::size_t value_rank() const
      {
	return 2;
      }
    /*!
     *  \brief value_dimension
     *
     *  This method evaluates 
     *
     */
    virtual std::size_t value_dimension(uint i) const
      {
	return 2;
      }
  };
}
#endif
