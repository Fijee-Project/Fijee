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
#include "Cell_conductivity.h"
//
// We give a comprehensive type name
//
typedef Domains::Cell_conductivity DCc;
//
//
//
DCc::Cell_conductivity():
  cell_id_(0), cell_subdomain_(0)
{    
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = 0.;
  //
  for ( int i = 0 ; i < 3 ; i++ )
    centroid_lambda_[i] = Domains::Point_vector();
}
//
//
//
DCc::Cell_conductivity( int Cell_id, int Cell_subdomain, 
			float Centroid_x, float Centroid_y, float Centroid_z, 
			float L1, float V1_x, float V1_y, float V1_z, 
			float L2, float V2_x, float V2_y, float V2_z,
			float L3, float V3_x, float V3_y, float V3_z,
			float C00, float C01, float C02, float C11, float C12, float C22 ):
  cell_id_(Cell_id), cell_subdomain_(Cell_subdomain)
{    
    conductivity_coefficients_[0] = C00;
    conductivity_coefficients_[1] = C01;
    conductivity_coefficients_[2] = C02;
    conductivity_coefficients_[3] = C11;
    conductivity_coefficients_[4] = C12;
    conductivity_coefficients_[5] = C22;
    //
    centroid_lambda_[0] = Domains::Point_vector( Centroid_x, Centroid_y, Centroid_z, V1_x, V1_y, V1_z, L1);
    centroid_lambda_[1] = Domains::Point_vector( Centroid_x, Centroid_y, Centroid_z, V2_x, V2_y, V2_z, L2);
    centroid_lambda_[2] = Domains::Point_vector( Centroid_x, Centroid_y, Centroid_z, V3_x, V3_y, V3_z, L3);
}
//
//
//
DCc::Cell_conductivity( const DCc& that ):
  cell_id_(that.cell_id_), cell_subdomain_(that.cell_subdomain_)
{
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.conductivity_coefficients_[i];
  //
  for ( int i = 0 ; i < 3 ; i++ )
    centroid_lambda_[i] = that.centroid_lambda_[i];
}
//
//
//
DCc::~Cell_conductivity()
{
}
//
//
//
DCc& 
DCc::operator = ( const DCc& that )
{
  cell_id_        = that.cell_id_;
  cell_subdomain_ = that.cell_subdomain_;
  //
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.conductivity_coefficients_[i];
  //
  for ( int i = 0 ; i < 3 ; i++ )
    centroid_lambda_[i] = that.centroid_lambda_[i];

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		      const DCc& that)
{
//  std::for_each( that.get_list_position().begin(),
//		 that.get_list_position().end(),
//		 [&stream]( int Val )
//		 {
//		   stream << "list pos = " << Val << "\n";
//		 });
//  //
//  stream << "position x = " <<    that.get_pos_x() << "\n";
//  stream << "position y = " <<    that.get_pos_y() << "\n";
//  if ( &that.get_tab() )
//    {
//      stream << "tab[0] = "     << ( &that.get_tab() )[0] << "\n";
//      stream << "tab[1] = "     << ( &that.get_tab() )[1] << "\n";
//      stream << "tab[2] = "     << ( &that.get_tab() )[2] << "\n";
//      stream << "tab[3] = "     << ( &that.get_tab() )[3] << "\n";
//    }
  //
  return stream;
};
