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
#include "Electrodes_setup.h"
#include "Electrodes_surface.h"
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
    //! Electrodes list
    std::shared_ptr< Solver::Electrodes_setup > electrodes_;
    //! Head model facets collection
    std::shared_ptr< MeshValueCollection< std::size_t > > mesh_facets_collection_;
    //! Function space
    std::shared_ptr< tCS_model::FunctionSpace > V_;
    //! Function space
    std::shared_ptr< tCS_field_model::FunctionSpace > V_field_;
    //! Periphery
    std::shared_ptr< Periphery > perifery_;
    //! Boundarie conditions
    std::shared_ptr< MeshFunction< std::size_t > > boundaries_;
    //! Sample studied
    int sample_;

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
    ~tCS_tACS(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class tCS_tACS
     *
     */
    tCS_tACS& operator = ( const tCS_tACS& ){};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class tCS_tACS
     *
     */
    void operator ()();
    
  public:
    /*!
     *  \brief number of physical event
     *
     *
     */
    inline int get_number_of_physical_event(){return electrodes_->get_number_samples_(); };
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
  };
}
#endif
