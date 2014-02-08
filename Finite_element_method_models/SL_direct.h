#ifndef _SL_DIRECT_H
#define _SL_DIRECT_H
#include <list>
#include <memory>
#include <string>
#include <mutex>
#include <stdexcept>      // std::logic_error
//
// FEniCS
//
#include <dolfin.h>
// Source localization direct model
#include "SLD_model.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Source.h"
#include "Conductivity.h"
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
 * \file SL_direct.h
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
  /*! \class SL_direct
   * \brief classe representing the source localisation with direct method.
   *
   *  This class representing the Physical model for the source localisation using the direct method.
   */
  class SL_direct
  {
    //! Dipoles list
    std::list< Solver::Current_density > dipoles_list_;
    //! Head model mesh
    boost::shared_ptr< Mesh > mesh_;
    //! Head model sub domains
    boost::shared_ptr< MeshFunction< long unsigned int > > domains_;
    //! Anisotropic conductivity
    boost::shared_ptr< Solver::Tensor_conductivity > sigma_;
    //! Function space
    boost::shared_ptr< SLD_model::FunctionSpace > V_;
    //! Periphery
    boost::shared_ptr< Periphery > perifery_;
    //! Boundarie conditions
    boost::shared_ptr<  FacetFunction< size_t > > boundaries_;
    
  private:
    std::mutex critical_zone_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class SL_direct
     *
     */
    SL_direct();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    SL_direct( const SL_direct& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class SL_direct
     */
    ~SL_direct(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class SL_direct
     *
     */
    SL_direct& operator = ( const SL_direct& ){};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class SL_direct
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
