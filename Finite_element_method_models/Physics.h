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
     *  \brief XML output
     *
     *  This method generates XML output.
     *
     */
    virtual void XML_output() = 0;
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
