#ifndef _PHYSICS_H
#define _PHYSICS_H
#include <fstream>  
//
// FEniCS
//
#include <dolfin.h>
//
// UCSF project
//
#include "Utils/Fijee_environment.h"
#include "PDE_solver_parameters.h"
#include "Conductivity.h"
#include "Electrodes_setup.h"
#include "Electrodes_surface.h"

//using namespace dolfin;

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
//
// 
//
namespace Solver
{
  /*! \class Physics
   * \brief classe representing the mother class of all physical process: source localization (direct and subtraction), transcranial Current Stimulation (tDCS, tACS).
   *
   *  This class representing the Physical model.
   */
  class Physics
  {
  protected:
    //! Head model mesh
    std::shared_ptr< dolfin::Mesh > mesh_;
    //! Head model sub domains
    std::shared_ptr< MeshFunction< std::size_t > > domains_;
    //! Anisotropic conductivity
    std::shared_ptr< Solver::Tensor_conductivity > sigma_;
    //! Electrodes list
    std::shared_ptr< Solver::Electrodes_setup > electrodes_;
    //! Head model facets collection
    std::shared_ptr< MeshValueCollection< std::size_t > > mesh_facets_collection_;
    //! Boundarie conditions
    std::shared_ptr< MeshFunction< std::size_t > > boundaries_;


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Physics
     *
     */
    Physics();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Physics( const Physics& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Physics
     */
    virtual ~Physics(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Physics
     *
     */
    Physics& operator = ( const Physics& ){return *this;};

  public:
    /*!
     *  \brief Solution domain extraction
     *
     *  This method extract from the Function solution U the sub solution covering the sub-domains Sub_domains.
     *  The result is a file with the name tDCS_{Sub_domains}.vtu
     *
     *  \param U: Function solution of the Partial Differential Equation.
     *  \param Sub_domains: array of sub-domain we want to extract from U.
     *  \param name: file name.
     *
     */
    void solution_domain_extraction( const dolfin::Function&,  std::list<std::size_t>&, 
				     const char* );

  public:
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Physics
     *
     */
    virtual void operator ()() = 0;
    /*!
     *  \brief Get number of physical events
     *
     *  This method return the number of parallel process for the Physics solver. 
     *
     */
    virtual inline
      int get_number_of_physical_events() = 0;
   };
};

#endif
