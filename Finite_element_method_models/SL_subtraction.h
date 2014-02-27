#ifndef _SL_SUBTRACTION_H
#define _SL_SUBTRACTION_H
#include <list>
#include <memory>
#include <string>
#include <mutex>
#include <stdexcept>      // std::logic_error
//
// FEniCS
//
#include <dolfin.h>
// Source localization subtraction model
#include "SLS_model.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Physics.h"
#include "Source.h"
#include "Boundaries.h"
#include "Sub_domaines.h"
#include "PDE_solver_parameters.h"
//#include "Utils/Thread_dispatching.h"
//
//
//
//using namespace dolfin;
//
/*!
 * \file SL_subtraction.h
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
  /*! \class SL_subtraction
   * \brief classe representing the source localisation with subtraction method.
   *
   *  This class representing the Physical model for the source localisation using the subtraction method.
   */
  class SL_subtraction : Physics
  {
    //! Dipoles list
    std::list< Solver::Phi > dipoles_list_;
    //! Function space
    boost::shared_ptr< SLS_model::FunctionSpace > V_;
    //! Boundarie conditions
    boost::shared_ptr<  FacetFunction< size_t > > boundaries_;
    
  private:
    std::mutex critical_zone_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class SL_subtraction
     *
     */
    SL_subtraction();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    SL_subtraction( const SL_subtraction& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class SL_subtraction
     */
    ~SL_subtraction(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class SL_subtraction
     *
     */
    SL_subtraction& operator = ( const SL_subtraction& ){return *this;};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class SL_subtraction
     *
     */
    void operator () ();
    
  public:
    /*!
     */
    inline
      int get_number_of_physical_event(){return number_dipoles_; };


  private:
    /// Number of dipoles
    int number_dipoles_;

  };
}
#endif
