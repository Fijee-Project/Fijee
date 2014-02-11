#ifndef TCS_TDCS_H
#define TCS_TDCS_H
#include <list>
#include <memory>
#include <string>
#include <mutex>
#include <stdexcept>      // std::logic_error
#include <map>
//
// FEniCS
//
#include <dolfin.h>
// transcranial current stimulation
#include "tCS_model.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Electrodes_setup.h"
#include "Electrodes_surface.h"
#include "Conductivity.h"
#include "Boundaries.h"
#include "Sub_domaines.h"
#include "PDE_solver_parameters.h"
//#include "Utils/Thread_dispatching.h"
//
//
//
typedef std::vector<std::vector<std::pair<dolfin::la_index, dolfin::la_index> > > Global_dof_to_cell_dof;
//
//
//
/*!
 * \file tCS_tDCS.h
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
  /*! \class tCS_tDCS
   * \brief classe representing the source localisation with direct method.
   *
   *  This class representing the Physical model for the source localisation using the direct method.
   */
  class tCS_tDCS
  {
    //! Electrodes list
    boost::shared_ptr< Solver::Electrodes_setup > electrodes_;
    //! Head model mesh
    boost::shared_ptr< Mesh > mesh_;
    //! Head model facets collection
    boost::shared_ptr< MeshValueCollection< std::size_t > > mesh_facets_collection_;
    //! Head model sub domains
    boost::shared_ptr< MeshFunction< long unsigned int > > domains_;
    //! Anisotropic conductivity
    boost::shared_ptr< Solver::Tensor_conductivity > sigma_;
    //! Function space
    boost::shared_ptr< tCS_model::FunctionSpace > V_;
    //! Periphery
    boost::shared_ptr< Periphery > perifery_;
    //! Boundarie conditions
    boost::shared_ptr< MeshFunction< std::size_t > > boundaries_;

  std::map< std::size_t, std::size_t > map_index_cell_;

    
  private:
    std::mutex critical_zone_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class tCS_tDCS
     *
     */
    tCS_tDCS();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    tCS_tDCS( const tCS_tDCS& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class tCS_tDCS
     */
    ~tCS_tDCS(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class tCS_tDCS
     *
     */
    tCS_tDCS& operator = ( const tCS_tDCS& ){};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class tCS_tDCS
     *
     */
    void operator () ();
    
  public:
    /*!
     *  \brief number of physical event
     *
     *
     */
    inline
      int get_number_of_physical_event(){return 1; };
    /*!
     *  \brief function filter
     *
     *
     */
    void function_filter(const Function& u, Function& u_filtered, std::size_t );
    /*!
     *  \brief function filter
     *
     *
     */
    void output_filter(const Function& u, std::size_t );

  private:
    /*!
     *  \brief 
     *
     *
     */
    void build_global_to_cell_dof(Global_dof_to_cell_dof&, const FunctionSpace& );

  };
}
#endif
