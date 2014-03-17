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
    std::shared_ptr< Solver::Electrodes_setup > electrodes_;
    //! Boundary mesh function
    std::shared_ptr< MeshFunction< std::size_t > > boundaries_;
    //! 
    mutable std::list< std::tuple< 
      std::string,  // 0 - electrode label
      Point,        // 1 - vertex (can't take vertex directly: no operator =)
      MeshEntity,   // 2 - facet
      std::size_t,  // 3 - cell index (tetrahedron)
      std::size_t,  // 4 - vertex index (-1 for midpoint)
      bool>         // 5 - criteria satisfaction
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
    Electrodes_surface( std::shared_ptr< Solver::Electrodes_setup > ,
			const std::shared_ptr< MeshFunction< std::size_t > > ,
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
     *  \brief Surface vertices per electrodes
     *
     *  This method tracks the cells on the boundaries. For each electrodes_ it provides subdomain boundary information. 
     *
     */
    void surface_vertices_per_electrodes( const std::size_t );

  private:
    /*!
     *  \brief 
     *
     *  This method return true for points (Array) inside the subdomain.
     *
     * \param x (_Array_ <double>): The coordinates of the point.
     * \param on_boundary: True if x is on the boundary
     *
     * \return True for points inside the subdomain.
     *
     */
    virtual bool inside(const Array<double>& , bool ) const;
  };
}
#endif
