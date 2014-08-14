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
#ifndef FIJEE_LOG_MANAGEMENT_H
#define FIJEE_LOG_MANAGEMENT_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Fijee_log_management.h
 * \brief Log management 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <errno.h>      /* builtin errno */
#include <cstring>      /* strerror */
#include <exception>
// steady_clock example
#include <ctime>
#include <ratio>
#include <chrono>
#include <fstream>      // std::filebuf
//
// UCSF
//
#include "Fijee_environment.h"
//
//
//
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Time_profiler
   * \brief classe representing time profiling
   *
   *  This class is the time profiling class.
   */
  class Time_profiler
  {
  private:
    //! Starting point
    std::chrono::steady_clock::time_point starting_point_;
    //! Name of the function
    std::string function_name_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Time_profiler
     *
     */
  Time_profiler( const std::string& Function_name):
    starting_point_(std::chrono::steady_clock::now() ), 
      function_name_(Function_name){};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Time_profiler( const Time_profiler& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Time_profiler
     */
    ~Time_profiler()
      {
	// End the time profiling
 	std::chrono::steady_clock::time_point ending_point = std::chrono::steady_clock::now();
	// compute elaps time
	std::chrono::duration<double> time_span = 
	  std::chrono::duration_cast< std::chrono::duration<double> >(ending_point-starting_point_);

	// 
	// output stream the result
	std::ofstream ofs ("fijee_time.log", std::ofstream::app );
	// stream and dump the result in a file
	ofs << "[Fijee time profiler] Function: " << function_name_
	    << " lasted " << time_span.count() << " seconds."
	    << std::endl;
	//
	ofs.close();
     };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Time_profiler
     *
     */
    Time_profiler& operator = ( const Time_profiler& ){return *this;};
  };

#ifdef TIME_PROFILER
#  define FIJEE_TIME_PROFILER(NAME) \
  Utils::Time_profiler timer( std::string(NAME) );
#else
#  define FIJEE_TIME_PROFILER(NAME)
#endif
}
#endif
