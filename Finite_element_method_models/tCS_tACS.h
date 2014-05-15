#ifndef TCS_TACS_H
#define TCS_TACS_H
#include <list>
#include <memory>
#include <string>
#include <mutex>
#include <stdexcept>      // std::logic_error
#include <map>
#include <thread>         // std::thread
//
// FEniCS
//
#include <dolfin.h>
// transcranial current stimulation
#include "tCS_model.h"
#include "tCS_field_model.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Physics.h"
#include "Conductivity.h"
#include "Boundaries.h"
#include "Sub_domaines.h"
#include "PDE_solver_parameters.h"
//#include "Utils/Thread_dispatching.h"
// Validation
#include "Spheres_electric_monopole.h"
//
//
//
typedef std::vector<std::vector<std::pair<dolfin::la_index, dolfin::la_index> > > Global_dof_to_cell_dof;
//
//
//
/*!
 * \file tCS_tACS.h
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
  /*! \class tCS_tACS
   * \brief classe representing tACS simulation.
   *
   *  This class representing the Physical model for the transcranial Alternating Current Stimulation (tACS) simulation.
   */
  class tCS_tACS : Physics
  {
  private:
    //! Function space
    std::shared_ptr< tCS_model::FunctionSpace > V_;
    //! Function space
    std::shared_ptr< tCS_field_model::FunctionSpace > V_field_;
    //! Sample studied
    int sample_;
    // Head time series potential output file
    File *file_potential_time_series_;
    // Brain time series potential output file
    File *file_brain_potential_time_series_;
    // Time series potential field output file
    File *file_field_time_series_;

    // std::map< std::size_t, std::size_t > map_index_cell_;

    
  private:
    std::mutex critical_zone_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class tCS_tACS
     *
     */
    tCS_tACS();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    tCS_tACS( const tCS_tACS& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class tCS_tACS
     */
    virtual ~tCS_tACS(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class tCS_tACS
     *
     */
    tCS_tACS& operator = ( const tCS_tACS& ){return *this;};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class tCS_tACS
     *
     */
    virtual void operator ()();
    
  public:
    /*!
     *  \brief Get number of physical events
     *
     *  This method return the number of parallel process for the Physics solver. 
     *  In the case of tACS the number of events is the number of sampling in the time series.
     *
     */
    virtual inline
      int get_number_of_physical_events(){return electrodes_->get_number_samples_();};
    /*!
     *  \brief Solution domain extraction
     *
     *  This method extract from the Function solution U the sub solution covering the sub-domains Sub_domains.
     *  The result is a file with the name tACS_{Sub_domains}.vtu
     *
     *  \param U: Function solution of the Partial Differential Equation.
     *  \param Sub_domains: array of sub-domain we want to extract from U.
     *
     */
    void regulation_factor(const Function& , std::list<std::size_t>& );
    /*!
     *  \brief Solution domain extraction
     *
     *  This method extract from the Function solution U the sub solution covering the sub-domains Sub_domains.
     *  The result is a file with the name tDCS_{Sub_domains}.vtu
     *
     *  \param U: Function solution of the Partial Differential Equation.
     *  \param Sub_domains: array of sub-domain we want to extract from U.
     *  \param File: file .
     *
     */
    void solution_domain_extraction( const dolfin::Function&, const std::list<std::size_t>&, 
				     File* );

  };
}
#endif
