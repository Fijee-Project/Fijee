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
