#ifndef _CONDUCTIVITY_H
#define _CONDUCTIVITY_H
#include <dolfin.h>
#include <vector>
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


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Tensor_conductivity
     *
     */
    Tensor_conductivity( boost::shared_ptr< const Mesh > );
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
