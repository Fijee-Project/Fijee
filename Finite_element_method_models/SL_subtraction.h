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
#ifndef _SL_SUBTRACTION_H
#define _SL_SUBTRACTION_H
#include <list>
#include <memory>
#include <string>
#include <mutex>
#include <stdexcept>      // std::logic_error
//
// FEniCS
//
#include <dolfin.h>
// Source localization subtraction model
#include "SLS_model.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Physics.h"
#include "Boundaries.h"
#include "Source.h"
#include "PDE_solver_parameters.h"
//
//
//
//using namespace dolfin;
//
/*!
 * \file SL_subtraction.h
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
  /*! \class SL_subtraction
   * \brief classe representing the source localisation with subtraction method.
   *
   *  This class representing the Physical model for the source localisation using the subtraction method.
   */
  class SL_subtraction : Physics
  {
  private:
    //! Number of dipoles
    int number_dipoles_;
    //! Dipoles list
    std::list< Solver::Phi > dipoles_list_;
    //! Function space
    std::shared_ptr< SLS_model::FunctionSpace > V_;
    //! Bilinear form
    std::shared_ptr< SLS_model::BilinearForm > a_;
    //! Assembling matrix of the Bilinear form
    std::unique_ptr< Matrix > A_;
    //! Initializing variable check: we want A_ to be built one time
    bool initialized_;
    
  private:
    std::mutex critical_zone_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class SL_subtraction
     *
     */
    SL_subtraction();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    SL_subtraction( const SL_subtraction& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class SL_subtraction
     */
    virtual ~SL_subtraction(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class SL_subtraction
     *
     */
    SL_subtraction& operator = ( const SL_subtraction& ){return *this;};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class SL_subtraction
     *
     */
    virtual void operator () ();
    
  public:
    /*!
     *  \brief Get number of physical events
     *
     *  This method return the number of parallel process for the Physics solver. 
     *  In the case of source localization the number of events is the number of dipoles simulated.
     *
     */
    virtual inline
      int get_number_of_physical_events(){return number_dipoles_;};
    /*!
     *  \brief XML output
     *
     *  This method generates XML output.
     *
     */
    virtual void XML_output(){};
  };
}
#endif
