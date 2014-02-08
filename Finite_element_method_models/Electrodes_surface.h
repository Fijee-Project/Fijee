#ifndef _ELECTRODES_SURFACE_H
#define _ELECTRODES_SURFACE_H
#include <dolfin.h>
#include <vector>
#include <tuple> 
//
// UCSF
//
#include "PDE_solver_parameters.h"
#include "Electrodes_setup.h"
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
  /*! \class Electrodes_surface
   * \brief classe representing anisotrope conductivity
   *
   *  This class is an example of class I will have to use
   */
  class Electrodes_surface : public SubDomain
  {
  private:
    //! Electrodes setup
    boost::shared_ptr< Solver::Electrodes_setup > electrodes_;
    //! Boundary mesh function
    boost::shared_ptr< MeshFunction< std::size_t > > boundaries_;
    //! 
    mutable std::list< std::tuple< 
    std::string, // electrode label
      Point,        // vertex (can't take vertex directly: no operator =)
      std::size_t,  // vertex index (mesh)
      std::size_t,  // cell index (tetrahedron)
      bool,         // vertex: true ; midpoint: false
      bool>         // criteria satisfaction
      > list_vertices_;
  
  
 public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_surface
     *
     */
 Electrodes_surface():SubDomain(){};
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Electrodes_surface
     *
     */
    Electrodes_surface( const boost::shared_ptr< Solver::Electrodes_setup > ,
			const boost::shared_ptr< MeshFunction< std::size_t > > ,
			const std::map< std::size_t, std::size_t >&  );
    /*!
     *  \brief destructor
     *
     *  Constructor of the class Electrodes_surface
     *
     */
    ~Electrodes_surface(){/*Do nothing*/};

  public:
    /*!
     *  \brief 
     *
     * 
     *
     */
    void surface_vertices_per_electrodes();

  private:
    /*!
     *  \brief 
     *
     * 
     *
     */
    virtual bool inside(const Array<double>& , bool ) const;
  };
}
#endif
