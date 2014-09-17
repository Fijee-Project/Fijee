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
#ifndef _ELECTRODES_SETUP_H
#define _ELECTRODES_SETUP_H
#include <dolfin.h>
#include <vector>
//
// FEniCS
//
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
#include "Utils/XML_writer.h"
//
// UCSF project
//
#include "Utils/Fijee_environment.h"
#include "Electrodes_injection.h"
#include "Conductivity.h"
#include "Intensity.h"
#include "PDE_solver_parameters.h"
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
  /*! \class Electrodes_setup
   * \brief classe representing the set up of electrodes
   *
   *  This class holds the set of electrodes for each time's setp of the electrodes measure. Electrodes_setup is the interface between the electrodes and the Physical models. 
   */
  class Electrodes_setup: public Utils::XML_writer
  {
  private:
    //! Electrodes list for current injected
    std::vector< std::shared_ptr< Solver::Electrodes_injection > > current_setup_;
//    //! Electrodes list for current injected
//    std::shared_ptr< Solver::Electrodes_injection > current_injection_;
    //! number of samples
    int number_samples_;
    //! number of electrodes
    int number_electrodes_;

    // 
    // Output file
    //! XML output file: setup node
    pugi::xml_node setup_node_;
    //! XML output file: electrodes node
    pugi::xml_node electrodes_node_;
    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    Electrodes_setup();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    Electrodes_setup(const Electrodes_setup& ){};
    /*!
     *  \brief Destructor
     *
     *  Constructor of the class Electrodes_setup
     *
     */
    virtual ~Electrodes_setup(){/*Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Electrodes_setup
     *
     */
    Electrodes_setup& operator = (const Electrodes_setup& ){return *this;};
    /*!
     *  \brief Operator []
     *
     *  Operator [] the class Electrodes_setup
     *
     */
    //    const Solver::Intensity& operator [] (const char * label )const{return get_current()->information(label);};

  public:
    /*!
     *  \brief Get number samples
     *
     *  This method return the number_samples_ member.
     *
     */
    ucsf_get_macro( number_samples_, int );
    /*!
     *  \brief Get number electrodes
     *
     *  This method return the number_electrodes_ member.
     *
     */
    ucsf_get_macro( number_electrodes_, int );
    /*!
     *  \brief Get the current set up
     *
     *  This method return the current set up in electrodes for the sampling Sample.
     *
     *  \param Sample: sample selected from the electrode measures
     *
     */
    std::shared_ptr< Solver::Electrodes_injection > get_current(const int Sample ) const 
      { return current_setup_[Sample];};

  public:
    /*!
     *  \brief Inside
     *
     *   This method 
     *
     */
    bool inside( const Point& ) const;
    /*!
     *  \brief Add electrical potential
     *
     *   This method 
     *
     */
    bool add_potential_value( const Point&, const double );
    /*!
     *  \brief Add electrical potential
     *
     *   This method 
     *
     */
    bool add_potential_value( const std::string, const double );
    /*!
     *  \brief Inside probe
     *
     *  This method 
     *
     */
    std::tuple<std::string, bool> inside_probe( const Point& ) const;
     /*!
     *  \brief Set boundary cells
     *
     *  This method record the cell index per probes.
     *
     */
    void set_boundary_cells( const std::map< std::string, std::map< std::size_t, std::list< MeshEntity  >  >  > & );
    /*!
     *  \brief 
     *
     *  This method 
     *
     */
    void record_potential( int, int );
 };
}
#endif
