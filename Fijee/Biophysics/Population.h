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
#ifndef POPULATION_H
#define POPULATION_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Population.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <list>
#include <tuple>
/*! \class Population
 * \brief class representing whatever
 *
 *  This class is an example of class I will have to use
 */
/*! \namespace Biophysics
 * 
 * Name space for our new package
 *
 */
namespace Biophysics
{
  class Population
  {
  private:
    //! Position of the population
    double position_[3];
    //! Direction of the dipole
    double direction_[3];
    //! Intensity of the population
    double I_;
    //! Index population
    int index_;
    //! Index cell which belongs the population
    int index_cell_;
    //! Index parcel which belongs the population
    int index_parcel_;
    //! Isotrope conductivity of the cell which belongs the population
    double lambda_[3];
    //! Potential time series
    std::list< std::tuple< double/*time*/, double/*V*/ > > V_time_series_;
      

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Population
     *
     */
    Population();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Population
     *
     */
    Population( int, double, double, double, double, double, double,
		double, int, int,
		double, double, double );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Population( const Population& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Population( Population&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Population
     */
    virtual ~Population(){/* Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Population
     *
     */
    Population& operator = ( const Population& );
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Population
     *
     */
    Population& operator = ( Population&& );


  public:
    const double* get_position_()  const {return position_; }
    const double* get_direction_() const {return direction_;}
    double get_I_() const {return I_;}
    int get_index_() const {return index_;}
    int get_index_cell_() const {return index_cell_;}
    int get_index_parcel_() const {return index_parcel_;}
    const double* get_lambda_() const {return lambda_;}
    const std::list< std::tuple< double, double > >& get_V_time_series_() const 
    {return V_time_series_;}
    // 
    void set_V_time_series(std::list< std::tuple< double, double > >&& Vts)
    { 
      V_time_series_.clear();
      V_time_series_ = std::move( Vts );
    }

  };
  /*!
   *  \brief Dump values for Population
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Population : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Population& );
}
#endif
