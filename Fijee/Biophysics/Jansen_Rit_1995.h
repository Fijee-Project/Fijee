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
#ifndef JANSEN_RIT_1995_H
#define JANSEN_RIT_1995_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Jansen_Rit_1995.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <stdio.h>
#include <cmath> 
#include <ctgmath>
#include <random>
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>      // std::logic_error
//
// UCSF
//
#include "Brain_rhythm.h"
// 
// 
// 
/*! \namespace Biophysics
 * 
 * Name space for our new package
 *
 */
namespace Biophysics
{
  /*! \class Jansen_Rit_1995
   * \brief class representing the first cortical activity model
   *
   *  This class implement the Jansen and Rit model. It is the first mathematical cortical activity model, representing alpha waves. 
   * 
   */
  class Jansen_Rit_1995 : public Brain_rhythm
  {
  private:
    //! Random engine generator
    std::default_random_engine generator_;
    //! Normal distribution
    std::normal_distribution<double> distribution_;
    //! Number of pulse per second (random noise)
    double pulse_;

    // 
    // Parameter of the nonlinear sigmoid function, transforming the average membrane potential 
    // into an average density of action potential
    // 

    //! Determines the maximum firing rate of the neural population
    double e0_;
    //! Steepness of the sigmoidal transformation
    double r_;
    //! Postsynaptic potential for which a 50 % firing rate is achieved
    double v0_;
    // 
    // Synaptic contacts
    // 
    //! Average number of synaptic contacts
    double C_;
    //! Average number of synaptic contacts in the excitatory feedback loop
    double C1_;
    //! Average number of synaptic contacts in the excitatory feedback loop
    double C2_;
    //! Average number of synaptic contacts in the inhibitory feedback loop
    double C3_;
    //! Average number of synaptic contacts in the inhibitory feedback loop
    double C4_;

    // 
    // Spontaneous activity in a single-column model
    // 

    //! Membrane average time constant and dendritic tree average time delays
    double a_;
    //! Average excitatory synaptic gain
    double A_;
    //! Membrane average time constant and dendritic tree average time delays
    double b_;
    //! Average inhibitory synaptic gain
    double B_;
    //! White noise
    std::vector< double > p_;
     
    // 
    // Multi-threading
    // 
    //! Critical zone
    std::mutex critical_zone_;
    //! Population treated
    int population_;
    // ! Population treated localy in the thread
    // Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
    // static thread_local int local_population_;


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Jansen_Rit_1995
     *
     */
    Jansen_Rit_1995();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Jansen_Rit_1995( const Jansen_Rit_1995& ) = default;
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Jansen_Rit_1995
     *
     */
    Jansen_Rit_1995& operator = ( const Jansen_Rit_1995& ) = default;
    /*!
     *  \brief Destructor
     *
     *  Operator destructor of the class Jansen_Rit_1995
     *
     */
    virtual ~Jansen_Rit_1995(){/* Do nothing */};  
    /*!
     *  \brief Operator ()
     *
     *  Operator () of the class SL_subtraction
     *
     */
    virtual void operator () ( const Pop_to_elec_type );
     
  public:
    /*!
     *  \brief Initialization
     *
     *  This member function initializes the containers.
     *
     */
    virtual void init();
    /*!
     *  \brief  Brain rhythm modelization
     *
     *  This member function build an alpha rhythm based on Jansen Rit (1995) mathematical model.
     *
     */
    virtual void modelization();
    /*!
     *  \brief  Brain rhythm modelization at electrodes
     *
     *  This member function build an alpha rhythm at electrodes based on Jansen Rit (1995) mathematical model.
     *
     */
    virtual void modelization_at_electrodes();

  private:
    /*!
     *  \brief  Sigmoid
     *
     *  This member function transforms the average membrane potential into an average density of action potential.
     *
     */
    inline double sigmoid( const double V ){return (2 * e0_) / ( 1 + exp( r_*(v0_ - V) ) ); }


  public:
    /*!
     *  \brief  Ordinary differential equations (ODE)
     *
     *  This member function solves the Ordinary differential equations system using several type of algorythms.
     *
     *  \param t: time step. 
     *  \param Y: solution of the ODE system.
     *  \param DyDt: time derivative equation.
     *
     */
    int ordinary_differential_equations(double T, const double Y[], double DyDt[]);
  };
}
#endif
