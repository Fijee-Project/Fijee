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
