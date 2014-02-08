#ifndef _ELECTRODES_SETUP_H
#define _ELECTRODES_SETUP_H
#include <dolfin.h>
#include <vector>
//
// FEniCS
//
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Electrodes_injection.h"
#include "Conductivity.h"
//#include "Boundaries.h"
//#include "Sub_domaines.h"
#include "PDE_solver_parameters.h"
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
  /*! \class Electrodes_setup
   * \brief classe representing setup of electrodes
   *
   *  This class is an example of class I will have to use
   */
  class Electrodes_setup
  {
  private:
    //! Electrodes list for current injected
    boost::shared_ptr< Solver::Electrodes_injection > current_injection_;
    //! Electrodes list for potential applied
    boost::shared_ptr< Solver::Electrodes_injection > potential_injection_;
    //! number of electrodes
    int number_electrodes_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    Electrodes_setup();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    Electrodes_setup(const Electrodes_setup& ){};
    /*!
     *  \brief Destructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    ~Electrodes_setup(){/*Do nothing*/};
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    Electrodes_setup& operator = (const Electrodes_setup& ){return *this;};

  public:
    /*!
     *  \brief 
     *
     *  
     *
     */
    boost::shared_ptr< Solver::Electrodes_injection > get_current() const { return current_injection_;};
    /*!
     *  \brief 
     *
     *  
     *
     */
    boost::shared_ptr< Solver::Electrodes_injection > get_potential() const { return potential_injection_;};
    /*!
     *  \brief 
     *
     *  
     *
     */
    bool inside( const Point& ) const;
    /*!
     *  \brief 
     *
     *  
     *
     */
    std::tuple<std::string, bool> inside_probe( const Point& ) const;
  };
}
#endif
