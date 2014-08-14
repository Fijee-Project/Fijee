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
//
// FEniCS
//
#include <dolfin.h>
//
//
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "PDE_solver_parameters.h"
#include "Utils/Fijee_exception_handler.h"
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
    Field( const Field& ){};
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
    Field& operator = ( const Field& ){return *this;};
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
	  throw Utils::Exit_handler( message, 
				     __LINE__, __FILE__ );
	}
    }
    catch( Utils::Exception_handler& err )
      {
	std::cerr << err.what() << std::endl;
      }
  }
}
#endif
