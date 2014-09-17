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
#include "Dipole.h"
//
// We give a comprehensive type name
//
typedef Domains::Dipole DD;
//
//
//
DD::Dipole():
  Domains::Point_vector(),
  cell_id_(0), cell_subdomain_(0), cell_parcel_(0)
{
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = 0.;
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = 0.;
}
//
//
//
DD::Dipole( const DD& that ):
  Domains::Point_vector(that),
  cell_id_(that.cell_id_), cell_subdomain_(that.cell_subdomain_), cell_parcel_(that.cell_parcel_)
{
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.conductivity_coefficients_[i];
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.conductivity_eigenvalues_[i];
}
//
//
//
DD::Dipole( const Domains::Cell_conductivity& that ):
  Domains::Point_vector( that.get_centroid_lambda_()[0] ),
  cell_id_( that.get_cell_id_() ), cell_subdomain_(that.get_cell_subdomain_()), 
  cell_parcel_(that.get_cell_parcel_())
{
  //
  // set the dipole intensity
  set_weight_( 0.000000001 );
  
  //
  //
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.get_conductivity_coefficients_()[i];

  //
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.get_centroid_lambda_()[i].weight();
  
}
//
//
//
DD::Dipole( const Point_vector& that_point, const Domains::Cell_conductivity& that ):
  Domains::Point_vector( that_point ),
  cell_id_( that.get_cell_id_() ), cell_subdomain_(that.get_cell_subdomain_()),
  cell_parcel_(that.get_cell_parcel_())
{
  //
  // set the dipole intensity
  set_weight_( 0.000000001 );
  
  //
  //
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.get_conductivity_coefficients_()[i];

  //
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.get_centroid_lambda_()[i].weight();
  
}
//
//
//
DD::~Dipole(){ /* Do nothing */}
//
//
//
DD& 
DD::operator = ( const DD& that )
{
  Domains::Point_vector::operator = (that);
  //
  cell_id_        = that.cell_id_;
  cell_subdomain_ = that.cell_subdomain_;
  cell_parcel_    = that.cell_parcel_;
  //
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.conductivity_coefficients_[i];
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.conductivity_eigenvalues_[i];

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DD& that)
{
  //
  // 
  Domains::Point_vector position_normal = static_cast<Domains::Point_vector> (that);
  // [mm] to [m]
  position_normal.x() *= 1.e-3;
  position_normal.y() *= 1.e-3;
  position_normal.z() *= 1.e-3;


  //
  //
  stream 
    // Position Direction
    << position_normal
    // Dipole intensity
    << "I=\"" << that.weight() << "\" "
    // Cell id
    << "index_cell=\"" << that.get_cell_id_() << "\" "
    // Parcel id
    << "index_parcel=\"" << that.get_cell_parcel_() << "\" "
    // Conductivity eigenvalues
    << "lambda1=\"" << that.lambda1() 
    << "\" lambda2=\"" << that.lambda2() 
    << "\" lambda3=\"" << that.lambda3() << "\" ";
  
  //
  //
  return stream;
}
