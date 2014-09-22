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
#ifndef FIELD_H
#define FIELD_H
#include <algorithm>
//
// FEniCS
//
#include <dolfin.h>
//
// UCSF project
//
#include "PDE_solver_parameters.h"
#include "Fijee/Fijee_exception_handler.h"
#include "Utils/Third_party/pugi/pugixml.hpp"
//
//
//
typedef Solver::PDE_solver_parameters SDEsp;
//
//
//
/*!
 * \file Field.h
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
  /*! \class Field
   * \brief class representing physical field value at a particular position
   *
   *  This class is an example of class I will have to use
   */
  template < int Dimension = 1 >
    class Field
    {
    private:
    //! index of the parcell
    int index_; 
    //! Position X of the parcel's centroid
    double x_; 
    //! Position Y of the parcel's centroid
    double y_; 
    //! Position Z of the parcel's centroid
    double z_;
    //! Direction X of the parcel's centroid
    double vx_; 
    //! Direction Y of the parcel's centroid
    double vy_; 
    //! Direction Z of the parcel's centroid
    double vz_;
    //! Current
    double I_; 
    //! index of the cell   
    int index_cell_; 
    //! index value of the parcell
    int index_parcel_;
    //! Conductivity eigen value on the main direction
    double lambda1_; 
    //! Conductivity eigen value on the transverse direction
    double lambda2_; 
    //! Conductivity eigen value on the transverse direction
    double lambda3_;
    //! 
    std::map</*time*/double, /*vector info*/std::vector<double> > local_field_values_;
		       
      
    public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Field
     *
     */
    Field( int, 
	   double, double, double,
	   double, double, double,
	   double, int, int,
	   double, double, double );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Field( const Field<Dimension>& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a move constructor
     *
     */
    Field( Field<Dimension>&& );
    /*!
     *  \brief Destructor
     *
     *  Destructor of Field class
     *
     */
    ~Field(){/*Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Field
     *
     */
    Field<Dimension>& operator = ( const Field<Dimension>& );
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Field
     *
     */
    Field<Dimension>& operator = ( Field<Dimension>&& );
  
    public:
    //! index of the parcell
    ucsf_get_macro(index_, int); 
    //! Position X of the parcel's centroid
    ucsf_get_macro(x_, double); 
    //! Position Y of the parcel's centroid
    ucsf_get_macro(y_, double); 
    //! Position Z of the parcel's centroid
    ucsf_get_macro(z_, double);
    //! Direction X of the parcel's centroid
    ucsf_get_macro(vx_, double); 
    //! Direction Y of the parcel's centroid
    ucsf_get_macro(vy_, double); 
    //! Direction Z of the parcel's centroid
    ucsf_get_macro(vz_, double);
    //! Current
    ucsf_get_macro(I_, double); 
    //! index of the cell   
    ucsf_get_macro(index_cell_, int); 
    //! index value of the parcell
    ucsf_get_macro(index_parcel_, int);
    //! Conductivity eigen value on the main direction
    ucsf_get_macro(lambda1_, double); 
    //! Conductivity eigen value on the transverse direction
    ucsf_get_macro(lambda2_, double); 
    //! Conductivity eigen value on the transverse direction
    ucsf_get_macro(lambda3_, double);
    //! 
    std::map</*time*/double, /*vector info*/std::vector<double> >& 
    get_local_field_values_(){return local_field_values_;};

    public:
     /*!
     *  \brief Record field information
     *
     *  This method extract from the Function solution the field information for a parcel
     *
     */
    void record( const Function&, const double );
  };
  // 
  // 
  // 
  template < int Dimension >
    Field< Dimension >::Field( int Index, 
		       double X, double Y, double Z,
		       double Vx, double Vy, double Vz,
		       double I, int Index_cell, int Index_parcel,
		       double Lambda1, double Lambda2, double Lambda3 ):
    index_(Index), x_(X), y_(Y), z_(Z), vx_(Vx), vy_(Vy), vz_(Vz), I_(I), 
    index_cell_(Index_cell), index_parcel_(Index_parcel), 
    lambda1_(Lambda1), lambda2_(Lambda2), lambda3_(Lambda3)    
  {
    // 
    // Ensure the field dimensionality
    try{
      if( Dimension > 3 || Dimension < 1 )
	{
	  std::string message = "You are asking for the dimension: " + std::to_string(Dimension);
	  message += std::string(". \nDimension for a field must be between 1 and 3.");
	  throw Fijee::Exit_handler( message, 
				     __LINE__, __FILE__ );
	}
    }
    catch( Fijee::Exception_handler& err )
      {
	std::cerr << err.what() << std::endl;
      }
  }
  // 
  // 
  // 
  template < int Dimension >
    Field< Dimension >::Field( const Field<Dimension>& that ):
    index_(that.index_), 
    x_(that.x_), y_(that.y_), z_(that.z_), 
    vx_(that.vx_), vy_(that.vy_), vz_(that.vz_), 
    I_(that.I_), index_cell_(that.index_cell_), index_parcel_(that.index_parcel_), 
    lambda1_(that.lambda1_), lambda2_(that.lambda2_), lambda3_(that.lambda3_),
    local_field_values_(that.local_field_values_)
  {}
  // 
  // 
  // 
  template < int Dimension >
    Field< Dimension >::Field( Field<Dimension>&& that ):
    index_(0), 
    x_(0.), y_(0.), z_(0.), vx_(0.), vy_(0.), vz_(0.), 
    I_(0.), index_cell_(0), index_parcel_(0), 
    lambda1_(0.), lambda2_(0.), lambda3_(0.)
  {
    // 
    // Initialization of the object
    local_field_values_ = std::move(that.local_field_values_);
    
    // 
    // Copy the other object information
    index_ = that.index_; 
    x_  = that.x_; 
    y_  = that.y_; 
    z_  = that.z_;
    vx_ = that.vx_; 
    vy_ = that.vy_; 
    vz_ = that.vz_;
    I_ = that.I_; 
    index_cell_   = that.index_cell_; 
    index_parcel_ = that.index_parcel_;
    lambda1_ = that.lambda1_; 
    lambda2_ = that.lambda2_; 
    lambda3_ = that.lambda3_;
    
    // 
    // initialization 
    that.index_ = 0; 
    that.x_  = 0.; 
    that.y_  = 0.; 
    that.z_  = 0.;
    that.vx_ = 0.; 
    that.vy_ = 0.; 
    that.vz_ = 0.;
    that.I_  = 0.; 
    that.index_cell_   = 0; 
    that.index_parcel_ = 0;
    that.lambda1_ = 0.; 
    that.lambda2_ = 0.; 
    that.lambda3_ = 0.;
  }
  // 
  // 
  // 
  template < int Dimension > Field<Dimension>& 
    Field< Dimension >::operator = ( const Field<Dimension>& that )
    {
      if ( this != &that )
	{
	  index_ = that.index_; 
	  x_  = that.x_; 
	  y_  = that.y_; 
	  z_  = that.z_;
	  vx_ = that.vx_; 
	  vy_ = that.vy_; 
	  vz_ = that.vz_;
	  I_ = that.I_; 
	  index_cell_   = that.index_cell_; 
	  index_parcel_ = that.index_parcel_;
	  lambda1_ = that.lambda1_; 
	  lambda2_ = that.lambda2_; 
	  lambda3_ = that.lambda3_;
	  local_field_values_ = that.local_field_values_;
 	}
      
      // 
      // 
      return *this;
    };
  // 
  // 
  // 
  template < int Dimension > Field<Dimension>& 
    Field< Dimension >::operator = ( Field<Dimension>&& that )
    {
      if ( this != &that )
	{
	  // 
	  // initialization 
	  index_ = 0; 
	  x_  = 0.; 
	  y_  = 0.; 
	  z_  = 0.;
	  vx_ = 0.; 
	  vy_ = 0.; 
	  vz_ = 0.;
	  I_  = 0.; 
	  index_cell_   = 0; 
	  index_parcel_ = 0;
	  lambda1_ = 0.; 
	  lambda2_ = 0.; 
	  lambda3_ = 0.;
	  local_field_values_.clear();

	  // 
	  // Copy information from the other object
	  index_ = that.index_; 
	  x_  = that.x_; 
	  y_  = that.y_; 
	  z_  = that.z_;
	  vx_ = that.vx_; 
	  vy_ = that.vy_; 
	  vz_ = that.vz_;
	  I_ = that.I_; 
	  index_cell_   = that.index_cell_; 
	  index_parcel_ = that.index_parcel_;
	  lambda1_ = that.lambda1_; 
	  lambda2_ = that.lambda2_; 
	  lambda3_ = that.lambda3_;
	  local_field_values_ = std::move(that.local_field_values_);

	  // 
	  // initialization 
	  that.index_ = 0; 
	  that.x_  = 0.; 
	  that.y_  = 0.; 
	  that.z_  = 0.;
	  that.vx_ = 0.; 
	  that.vy_ = 0.; 
	  that.vz_ = 0.;
	  that.I_  = 0.; 
	  that.index_cell_   = 0; 
	  that.index_parcel_ = 0;
	  that.lambda1_ = 0.; 
	  that.lambda2_ = 0.; 
	  that.lambda3_ = 0.;
	}
      
      // 
      // 
      return *this;
    };
  // 
  // 
  // 
  template < int Dimension > void
    Field< Dimension >::record( const Function& Physic_field, const double Time)
    {
      // 
      // 
      Array<double> field_value(Dimension);
      std::vector<double> field_value_vector(Dimension);

      // 
      // 
      Array<double> position(3);
      // 
      position[0] = x_;
      position[1] = y_;
      position[2] = z_;
     
      // 
      // 
      Physic_field.eval(field_value, position);
      // 
      field_value_vector[0] = field_value[0];
      field_value_vector[1] = field_value[1];
      field_value_vector[2] = field_value[2];
      // 
      local_field_values_.insert( std::make_pair(Time, field_value_vector) );
    }
}
#endif
