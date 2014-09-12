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
#ifndef EEG_SIMULATION_H
#define EEG_SIMULATION_H
/*!
 * \file EEG_simulation.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <list>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <stdexcept>      // std::logic_error
// 
// GSL
// 
#include <gsl/gsl_errno.h>
//#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_complex.h>
// GSL macros
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])
//
// UCSF
//
#include "Utils/pugi/pugixml.hpp"
#include "Utils/Compression/Fijee_compression.h"
#include "Utils/Biophysics/Population.h"
#include "Utils/Biophysics/Leadfield_matrix.h"
#include "Utils/Statistical_analysis.h"
#include "Utils/XML_writer.h"
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
    /*! \class EEG_simulation
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class EEG_simulation: public Utils::Statistical_analysis, public Utils::XML_writer
    {
    private:
      //! Vector of analyse time v.s. potential for each population
      std::vector< std::vector< Bytef > > population_rhythm_;
      //! Vector of populations
      std::vector< Population/*dipole*/ > populations_;
      //! Vector of electrodes holding dipoles influence
      std::vector< Leadfield_matrix > leadfield_matrix_;
      //! For each electrode (vector) we have a list of alpha rhythm
      std::vector< std::vector< std::tuple< double/*time*/, double/*V*/ > > > 
	brain_rhythm_at_electrodes_;
      //! Number of electrodes
      int number_samples_;
     
      // 
      // Multi-threading
      // 
      //! Critical zone
      std::mutex critical_zone_;
      //! electrode treated
      int electrode_;
      // #! Electrode treated localy in the thread
      // Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
      // static thread_local int local_population_;



    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class EEG_simulation
       *
       */
      EEG_simulation();
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor of the class EEG_simulation
       *
       */
      EEG_simulation(const EEG_simulation& ) = delete;
      /*!
       *  \brief Destructor
       *
       *  Constructor of the class EEG_simulation
       *
       */
      virtual ~EEG_simulation(){/* Do nothing */};  
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class EEG_simulation
       *
       */
      EEG_simulation& operator = (const EEG_simulation& ) = delete;
      
    public:
      /*!
       *  \brief Load stimulation files
       *
       *  This method load the input XML files of alpha_rhythm per population (dipole) and the leadfield matrix.
       *
       * \param In_population_file_XML: input files in XML format.
       */
      void load_files( const std::string, const int Number_of_alpha_files = 1 );
      /*!
       *  \brief Load transcranial stimulation file
       *
       *  This method load the input XML file of transcranial simulation setting.
       *
       * \param In_population_file_XML: input transcranial stimulation file in XML format.
       */
      void load_tCS_file( const std::string );
      /*!
       *  \brief Get the number of populations
       *
       *  This method return the number of populations. This methode is needed for the multi-threading dispatching.
       *
       */
      inline int get_number_of_physical_events(){return number_samples_;};


    public:
      /*!
       */
      virtual void operator ()();
      /*!
       *  \brief  Brain rhythm modelization at the electrodes
       *
       *  This member function build an alpha rhythm at the electrodes.
       *
       */
      virtual void modelization();
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
