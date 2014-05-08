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
#include "Utils/Fijee_environment.h"
#include "Electrodes_injection.h"
#include "Conductivity.h"
#include "Intensity.h"
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
    std::vector< std::shared_ptr< Solver::Electrodes_injection > > current_setup_;
    //! Electrodes list for current injected
    std::shared_ptr< Solver::Electrodes_injection > current_injection_;
    //! number of samples
    int number_samples_;
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
     *  \brief Operator =
     *
     *  Operator = of the class Electrodes_setup
     *
     */
    Electrodes_setup& operator = (const Electrodes_setup& ){return *this;};
    /*!
     *  \brief Operator []
     *
     *  Operator [] the class Electrodes_setup
     *
     */
    //    const Solver::Intensity& operator [] (const char * label )const{return get_current()->information(label);};

  public:
    /*!
     *  \brief Get number samples
     *
     *  This method return the number_samples_ member.
     *
     */
    ucsf_get_macro( number_samples_, int );
    /*!
     *  \brief Get number electrodes
     *
     *  This method return the number_electrodes_ member.
     *
     */
    ucsf_get_macro( number_electrodes_, int );
    /*!
     *  \brief Get the current set up
     *
     *  This method return the current set up in electrodes for the sampling Sample.
     *
     *  \param Sample: sample selected from the electrode measures
     *
     */
    std::shared_ptr< Solver::Electrodes_injection > get_current(const int Sample ) const 
      { return current_setup_[Sample];};

  public:
    /*!
     *  \brief Inside
     *
     *   This method 
     *
     */
    bool inside( const Point& ) const;
    /*!
     *  \brief Add electrical potential
     *
     *   This method 
     *
     */
    bool add_potential_value( const Point&, const double );
    /*!
     *  \brief Add electrical potential
     *
     *   This method 
     *
     */
    bool add_potential_value( const std::string, const double );
    /*!
     *  \brief Inside probe
     *
     *  This method 
     *
     */
    std::tuple<std::string, bool> inside_probe( const Point& ) const;
     /*!
     *  \brief Set boundary cells
     *
     *  This method record the cell index per probes.
     *
     */
    void set_boundary_cells( const std::map< std::string, std::map< std::size_t, std::list< MeshEntity  >  >  > & );
 };
}
#endif
