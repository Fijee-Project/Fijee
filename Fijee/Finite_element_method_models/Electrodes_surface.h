//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
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
