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
#ifndef _INVERSE_SOLVER_H
#define _INVERSE_SOLVER_H
#include <fstream>  
//
// UCSF project
//
#include "Utils/Fijee_environment.h"

//using namespace dolfin;

/*! \namespace Inverse
 * 
 * Name space for our new package
 *
 */
//
// 
//
namespace Inverse
{
  /*! \class Physics
   * \brief classe representing the mother class of all physical process: source localization (direct and subtraction), transcranial Current Stimulation (tDCS, tACS).
   *
   *  This class representing the Physical model.
   */
  class Inverse_solver
  {
  public:
//    /*!
//     *  \brief Default Constructor
//     *
//     *  Constructor of the class Inverse_solver
//     *
//     */
//    Inverse_solver(){};
//    /*!
//     *  \brief Copy Constructor
//     *
//     *  Constructor is a copy constructor
//     *
//     */
//    Inverse_solver( const Inverse_solver& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Inverse_solver
     */
    virtual ~Inverse_solver(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Inverse_solver
     *
     */
    Inverse_solver& operator = ( const Inverse_solver& ){return *this;};
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Inverse_solver
     *
     */
    virtual void operator ()() = 0;
    /*!
     *  \brief Get number of physical events
     *
     *  This method return the number of parallel process for the Inverse_solver solver. 
     *
     */
    virtual inline
      int get_number_of_physical_events() = 0;
   };
};

#endif
