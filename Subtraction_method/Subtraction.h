#ifndef _SUBTRACTION_H
#define _SUBTRACTION_H
#include <list>
#include <memory>
#include <string>
//
// FEniCS
//
#include <dolfin.h>
#include "Poisson.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Physical_model.h"
#include "Source.h"
#include "Conductivity.h"
#include "Boundaries.h"
#include "Sub_Domaines.h"
#include "PDE_solver_parameters.h"
#include "Utils/Thread_dispatching.h"
//
//
//
//using namespace dolfin;
//
/*!
 * \file Subtraction.h
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
  /*! \class Subtraction
   * \brief classe representing the dipoles distribution
   *
   *  This class is an example of class I will have to use
   */
  class Subtraction : public Physical_model
  {
    //! Dipoles list
    std::list< Solver::Phi > dipoles_list_;
    //! Head model mesh
    std::unique_ptr< Mesh > mesh_;
    //! Head model sub domains
    std::unique_ptr< MeshFunction< long unsigned int > > domains_;
    //! Anisotropic conductivity
    std::unique_ptr< Solver::Tensor_conductivity > sigma_;
    //! Function space
    std::unique_ptr< Poisson::FunctionSpace > V_;
    // Boundaries
    std::unique_ptr<  FacetFunction< size_t > > boundaries_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Subtraction
     *
     */
    Subtraction();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Subtraction( const Subtraction& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Subtraction
     */
    virtual ~Subtraction(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Subtraction
     *
     */
    Subtraction& operator = ( const Subtraction& ){};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Subtraction
     *
     */
    void operator () ( Solver::Phi& /*,
		       Poisson::FunctionSpace&,
		       FacetFunction< size_t >&*/);

  public:
    /*!
     */
    virtual void solver_loop();


  private:
    /// Number of dipoles
    int number_dipoles_;

  };
}
#endif
