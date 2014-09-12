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
#ifndef MOLAEE_ARDEKANI_WENDLING_2009_H
#define MOLAEE_ARDEKANI_WENDLING_2009_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Molaee_Ardekani_Wendling_2009.h
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
#include <map>
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
    /*! \class Molaee_Ardekani_Wendling_2009
     * 
     * \brief classe representing a cortical activity model
     *
     *  This class implement the Molaee-Ardekani, Benquet, Bartolomei, Wendling model. It is a mathematical cortical activity model, representing alpha waves. 
     * 
     */
    class Molaee_Ardekani_Wendling_2009 : public Brain_rhythm
    {
    private:
      //! Random engine generator
      std::default_random_engine generator_;
      //! Normal distribution
      std::normal_distribution<double> distribution_;
      //! Duration of the simulation (ms)
      int duration_;
      //! Number of impulse per second (random noise)
      double pulse_;

      // 
      // Parameter of the nonlinear sigmoid function, transforming the average membrane potential 
      // into an average density of action potential
      // 

      //! Determines the maximum firing rate of the neural population
      double e0P_, e0I1_, e0I2_;
      //! Steepness of the sigmoidal transformation
      double rP_, rI1_, rI2_;
      //! Postsynaptic potential for which a 50 % firing rate is achieved
      double v0P_, v0I1_, v0I2_;
      // 
      // Synaptic contacts
      // 
      //! Average number of synaptic contacts: type P  \rightarrow type P
      double CPP_;
      //! Average number of synaptic contacts: type P  \rightarrow type I
      double CPI1_;
      //! Average number of synaptic contacts: type P  \rightarrow type I'
      double CPI2_;
      //! Average number of synaptic contacts: type I  \rightarrow type P
      double CI1P_;
      //! Average number of synaptic contacts: type I  \rightarrow type I
      double CI1I1_;
      //! Average number of synaptic contacts: type I' \rightarrow type P
      double CI2P_;
      //! Average number of synaptic contacts: type I' \rightarrow type I
      double CI2I1_;
      //! Average number of synaptic contacts: type I' \rightarrow type I'
      double CI2I2_;

      // 
      // Spontaneous activity in a single-column model
      // 

      //! Membrane average time constant and dendritic tree average time delays (AMPA)
      double a_;
      //! Average excitatory synaptic gain (AMPA)
      double A_;
      //! Membrane average time constant and dendritic tree average time delays (GABA_{a,slow})
      double b_;
      //! Average inhibitory synaptic gain (GABA_{a,slow})
      double B_;
      //! Membrane average time constant and dendritic tree average time delays (GABA_{a,fast})
      double g_;
      //! Average inhibitory synaptic gain (GABA_{a,fast})
      double G_;
      //! White noise
      std::vector< double > p_;

      //! White noise
      std::vector< std::map<double,double> > time_;
     
      // 
      // Multi-threading
      // 
      //! Critical zone
      std::mutex critical_zone_;
      //! Population treated
      int population_;
      // #! Population treated localy in the thread
      // Untill gcc fix the bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55800
      // static thread_local int local_population_;


    public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Molaee_Ardekani_Wendling_2009
	 *
	 */
	Molaee_Ardekani_Wendling_2009();
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Molaee_Ardekani_Wendling_2009( const Molaee_Ardekani_Wendling_2009& ) = default;
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Molaee_Ardekani_Wendling_2009
	 *
	 */
	Molaee_Ardekani_Wendling_2009& operator = ( const Molaee_Ardekani_Wendling_2009& ) = default;
	/*!
	 *  \brief Destructor
	 *
	 *  Operator destructor of the class Molaee_Ardekani_Wendling_2009
	 *
	 */
	virtual ~Molaee_Ardekani_Wendling_2009(){/* Do nothing */};  
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
	 *  This member function build an alpha rhythm based on Wendling (2009) mathematical model.
 	 *
	 */
	virtual void modelization();
	/*!
	 *  \brief  Brain rhythm modelization at electrodes
	 *
	 *  This member function build an alpha rhythm at electrodes based on Wendling (2009) mathematical model.
 	 *
	 */
	virtual void modelization_at_electrodes();

    private:
      /*!
       *  \brief  Sigmoid P
       *
       *  This member function transforms the average membrane potential into an average density of action potential. This wave-to-pulse transforms the pyramidal (type P) average mambrane potential.
       *
       */
      inline double sigmoid_P( const double V ){return (2 * e0P_) / ( 1 + exp( rP_*(v0P_ - V) ) ); }
      /*!
       *  \brief  Sigmoid I
       *
       *  This member function transforms the average membrane potential into an average density of action potential. This wave-to-pulse transforms the inter-neurons mediated by GABA_{a,fast} (type I) average mambrane potential.
       *
       */
      inline double sigmoid_I1( const double V ){return (2 * e0I1_) / ( 1 + exp( rI1_*(v0I1_ - V) ) ); }
      /*!
       *  \brief  Sigmoid I,
       *
       *  This member function transforms the average membrane potential into an average density of action potential. This wave-to-pulse transforms the inter-neurons mediated by GABA_{a,slow} (type I') average mambrane potential.
       *
       */
      inline double sigmoid_I2( const double V ){return (2 * e0I2_) / ( 1 + exp( rI2_*(v0I2_ - V) ) ); }


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
}
#endif
