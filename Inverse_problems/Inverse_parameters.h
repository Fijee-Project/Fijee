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
#ifndef INVERSE_PARAMETERS_H
#define INVERSE_PARAMETERS_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Inverse_parameters.hh
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <stdlib.h>     /* getenv */
#include <string>
#include <errno.h>      /* builtin errno */
//
// UCSF
//
#include "Utils/Fijee_environment.h"
#include "Utils/Fijee_log_management.h"
#include "Utils/enum.h"
//
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 * 
 *
 */
namespace Inverse
{
  /*! \class Inverse_parameters
   *  \brief class representing whatever
   *
   *  This class provides for encapsulation of persistent state information. It also avoids the issue of which code segment should "own" the static persistent object instance. It further guarantees what mechanism is used to allocate memory for the underlying object and allows for better control over its destruction.
   */
  class Inverse_parameters
  {
  private:
    //! unique instance
    static Inverse_parameters* parameters_instance_;
//    //! Freesurfer path. This path allow the program to reach all files we will need during the execution.
    std::string files_path_;
    std::string files_path_output_;
    std::string files_path_result_;
    std::string files_path_measure_;

    //
    // Dispatching information
    //! Number of threads in the dispatching pool
    int number_of_threads_;

  protected:
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Inverse_parameters
     *
     */
    Inverse_parameters();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Inverse_parameters( const Inverse_parameters& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Inverse_parameters
     */
    virtual ~Inverse_parameters();
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Inverse_parameters
     *
     */
    Inverse_parameters& operator = ( const Inverse_parameters& );



  public:
    /*!
     *  \brief Get the singleton instance
     *
     *  This method return the pointer parameters_instance_
     *
     */
    static Inverse_parameters* get_instance();
    /*!
     *  \brief Get files_path_output_
     *
     *  This method return the output path
     *
     */
    ucsf_get_string_macro(files_path_output_);
    /*!
     *  \brief Get files_path_result_
     *
     *  This method return the result path
     *
     */
    ucsf_get_string_macro(files_path_result_);
    /*!
     *  \brief Get files_path_measure_
     *
     *  This method return the measure path
     *
     */
    ucsf_get_string_macro(files_path_measure_);
    /*!
     *  \brief Get number_of_threads_
     *
     *  This method return the number of threads dispatched
     *
     */
    ucsf_get_macro(number_of_threads_, int);
    /*!
     *  \brief Kill the singleton instance
     *
     *  This method kill the singleton parameters_instance pointer
     *
     */
    static void kill_instance();
    /*!
     *  \brief init parameters
     *
     *  This method initialize the simulation parameters.
     *
     */
    void init();

  };
  /*!
   *  \brief Dump values for Inverse_parameters
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Inverse_parameters& );
}
#endif
