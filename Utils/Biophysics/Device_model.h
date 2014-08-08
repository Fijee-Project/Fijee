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
#ifndef DEVICE_MODEL_H
#define DEVICE_MODEL_H
#include <string>
//
// UCSF project
//
#include "Utils/Thread_dispatching.h"
//
//
/*!
 * \file Device_model.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Biophysics
  {
   /*! \class Device_model
     * \brief classe representing the dipoles distribution
     *
     *  This class is an example of class I will have to use
     */
    template < typename  Device_simulation, int num_of_threads = 1 >
      class Device_model
      {
      private:
      Device_simulation device_activity_;//neural_polpulation_activity_;

      public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Device_model
	 *
	 */
	Device_model(){};
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Device_model( const Device_model& ){};
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Device_model
	 *
	 */
	Device_model& operator = ( const Device_model& ){};
	//    /*!
	//     *  \brief Operator ()
	//     *
	//     *  Operator () of the class Device_model
	//     *
	//     */
	//    void operator () ();

      public:
	/*!
	 *  \brief minimize function
	 *
	 *  This method launch the minimization algorithm
	 */
      void alpha_rhythm_at_electrodes( std::string Output_File )
	{
 	  //
	  // load populations file and initialize the number of samples
	  device_activity_.load_files( Output_File );

	  //
	  // Define the number of threads in the pool of threads
	  Utils::Thread_dispatching pool( num_of_threads );
	  
	  //
	  //
	  for( int physical_event = 0 ;
	       physical_event != device_activity_.get_number_of_physical_events() ; 
	       physical_event++ )
	    // Enqueue tasks
	    pool.enqueue( std::ref(device_activity_) );
	};
	/*!
	 *  \brief minimize function
	 *
	 *  This method launch the minimization algorithm
	 */
	void output()
	{
	  // 
	  // Generation of output
	  device_activity_.output_XML();
	};
      };
  }
}
#endif
