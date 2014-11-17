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
#ifndef ELECTRODES_TACS_H
#define ELECTRODES_TACS_H
/*!
 * \file Electrodes_tACS.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <tuple>
//
// UCSF
//
#include "Utils/Third_party/pugi/pugixml.hpp"
#include "Fijee/Fijee_statistical_analysis.h"
#include "Fijee/Fijee_XML_writer.h"
#include "Fijee/Fijee_enum.h"
// 
// 
// 
/*! \namespace Electrodes
 * 
 * Name space for our new package
 *
 */
namespace Electrodes
{
  /* TODO
   *
   * Gathering all electrode classes in this namespace
   * 
   */
  typedef struct Electrode_struct
  {
    int index_;
    // position
    double position_x_; /* m */
    double position_y_; /* m */
    double position_z_; /* m */
    // Direction
    double direction_vx_; 
    double direction_vy_; 
    double direction_vz_;
    // Label
    std::string label_; 
    // Intensity
    double I_;         /* Ampere */
    double nu_;        /* Frequency */
    double amplitude_; /* Oscillation amplitude */
    double phase_;     /* injection phase */
    // Potential
    double V_; /* Volt */
    // Impedance
    double Re_z_l_;
    double Im_z_l_;
    // Contact surface between Electrode and the scalp
    std::string type_;
    double surface_; /* m^2 */
    double radius_;  /* m */
    double height_;  /* m */
  } Electrode;
  /*! \class Electrodes_tACS
   * \brief classe representing whatever
   *
   *  This class is an example of class 
   * 
   */
  class Electrodes_tACS: public Fijee::Statistical_analysis, public Fijee::XML_writer
  {
  private:
    //! Set of electrodes 
    std::map< std::string/*label*/, Electrode > electrodes_;
    //! Total of positive injection
    double I_tot_plus_;
    //! Total of negative injection
    double I_tot_minus_;
    //! Time step of the time series
    double time_step_;
    //! Time the proces starts
    double time_start_;
    //! Time the process ends
    double time_end_;
    //! Time injection ramp up
    double ramp_up_;
    //! Time injection ramp down
    double ramp_down_;
    //! Time series
    std::map< 
      std::string /* electrode label */, 
      std::vector< std::tuple< double/* time */, double/* intensity */ > > 
      > intensity_time_series_;


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_tACS
     *
     */
    Electrodes_tACS();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_tACS
     *
     */
    Electrodes_tACS(const std::vector< std::tuple< std::string, double, double, double, double >  >,
		    const std::vector< std::tuple< std::string, double, double, double, double >  >,
		    const double, const double, const double, const double );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor of the class Electrodes_tACS
     *
     */
    Electrodes_tACS(const Electrodes_tACS& ){};
    /*!
     *  \brief Destructor
     *
     *  Constructor of the class Electrodes_tACS
     *
     */
    virtual ~Electrodes_tACS(){/* Do nothing */};  
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Electrodes_tACS
     *
     */
    Electrodes_tACS& operator = (const Electrodes_tACS& ){return *this;};
      
  private:
    /*!
     *  \brief Output XML
     *
     *  This member function create the XML output
     *
     */
     double ramping_process( const double );
   

  public:
    /*!
     *  \brief Output XML
     *
     *  This member function create the XML output
     *
     */
    virtual void output_XML( const std::string );
    /*!
     */
    virtual void Make_analysis();
  };
}
#endif
