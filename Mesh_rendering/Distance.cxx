#include "Distance.h"
//
// We give a comprehensive type name
//
typedef Domains::Distance DDi;
//
//
//
DDi::Distance()
{
}
//
//
//
DDi::Distance( const DDi& that )
{
}
//
//
//
DDi::~Distance()
{
}
//
//
//
DDi& 
DDi::operator = ( const DDi& that )
{
  return *this;
}
//
//
//
float 
DDi::transformed_distance(const Point_vector& P1, const Point_vector& P2) const {
  float distx = P1.x() - P2.x();
  float disty = P1.y() - P2.y();
  float distz = P1.z() - P2.z();
  //
  return distx * distx + disty * disty + distz * distz;
}

//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DDi& that)
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
