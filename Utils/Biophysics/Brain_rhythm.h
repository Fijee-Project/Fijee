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
#ifndef BRAIN_RHYTHM_H
#define BRAIN_RHYTHM_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Brain_rhythm.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <stdio.h>
#include <string>
#include <list>
#include <map>
#include <vector>
#include <tuple>
#include <cstring>
// 
// GSL
// 
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_fft_complex.h>
// GSL macros
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])
//
// UCSF
//
#include "Utils/Fijee_exception_handler.h"
#include "Utils/enum.h"
#include "Utils/XML_writer.h"
#include "Utils/Fijee_exception_handler.h"
#include "Utils/Compression/Fijee_compression.h"
#include "Utils/pugi/pugixml.hpp"
#include "Utils/Statistical_analysis.h"
#include "Utils/Biophysics/Population.h"
#include "Utils/Biophysics/Leadfield_matrix.h"
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
  /*! \namespace Biophysics
   * 
   * Name space for our new package
   *
   */
  namespace Biophysics
  {
    /*! \class Brain_rhythm
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class Brain_rhythm: public Utils::Statistical_analysis, public Utils::XML_writer
    {
    protected:
      //! Duration of the simulation (ms)
      int duration_;
      //! Vector of analyse time v.s. potential for each population
      std::vector< std::vector< Bytef > > population_rhythm_;
      //! Vector hoding the offset of a time series regarding the time axes
      std::vector< double > population_V_shift_;
      //! Vector neural population
      std::vector< Population > populations_;
      //! Vector of electrodes holding dipoles influence
      std::vector< Leadfield_matrix > leadfield_matrix_;
      //! For each electrode (vector) we have a list of alpha rhythm
      std::vector< double* > brain_rhythm_at_electrodes_;
      //! Number of neural populations
      int number_samples_;
      //! Number of electrodes
      int number_electrodes_;

      // 
      // Transcranial current stimulation (tCS)
      // 

      //! Number of parcels in the parcellation
      int number_parcels_;
      //! Vector of potential from electric field in each parcel
      std::vector< Population > parcellation_;

      // 
      // Multi-threading
      // 
 
      //! electrode treated
      int electrode_;
      // #! Electrode treated localy in the thread
      // Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
      // static thread_local int local_population_;

      // 
      // Output
      // 
      //! 
      std::string alpha_rhythm_output_file_;

    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Brain_rhythm
       *
       */
      Brain_rhythm( const int );
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor of the class Brain_rhythm
       *
       */
      Brain_rhythm(const Brain_rhythm& ) = delete;
      /*!
       *  \brief Destructor
       *
       *  Constructor of the class Brain_rhythm
       *
       */
      virtual ~Brain_rhythm(){/* Do nothing */};  
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Brain_rhythm
       *
       */
      Brain_rhythm& operator = (const Brain_rhythm& ) = delete;
      
    public:
      /*!
       *  \brief Load population file
       *
       *  This method load the input XML file of population setting.
       *
       * \param In_population_file_XML: input population file in XML format.
       */
      void load_population_file( std::string );
      /*!
       *  \brief Load leadfield matrix file
       *
       *  This method load the leadfield matrix XML file.
       *
       * \param In_population_file_XML: input  file in XML format.
       */
      void load_leadfield_matrix_file( std::string );
      /*!
       *  \brief Load electric field file
       *
       *  This method load the electric field XML file. This file hold the electric field for each parcel at the centroid of the parcel.
       *
       * \param In_population_file_XML: input  file in XML format.
       */
      void load_electric_field_file( std::string, const double );
     /*!
       *  \brief Get the number of populations
       *
       *  This method return the number of populations. This methode is needed for the multi-threading dispatching.
       *
       */
      inline int get_number_of_physical_events(){return number_samples_;};
     /*!
       *  \brief Get the number of electrodes
       *
       *  This method return the number of electrodes. This methode is needed for the multi-threading dispatching.
       *
       */
      inline int get_number_of_electrodes_(){return number_electrodes_;};


    public:
      /*!
       *  \brief Initialization
       *
       *  This member function initializes the containers.
       *
       */
      virtual void init() = 0;
      /*!
       */
      virtual void modelization() = 0;
      /*!
       */
      virtual void modelization_at_electrodes() = 0;
      /*!
       */
      virtual void operator ()( const Pop_to_elec_type ) = 0;
      /*!
       *  \brief Output XML at the electrodes
       *
       *  This member function create the XML output at the electrodes
       *
       */
      virtual void output_electrodes_XML();
      /*!
       *  \brief Output XML
       *
       *  This member function create the XML output
       *
       */
      virtual void output_XML();
      /*!
       */
      virtual void Make_analysis();
    };
  }
}
#endif
