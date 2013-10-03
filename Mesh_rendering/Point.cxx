#include "Point.h"
//
// We give a comprehensive type name
//
typedef Domains::Point DPo;
//
//
//
DPo::Point()
{
  position_[0] = position_[1] = position_[2] = 0.;
}
//
//
//
DPo::Point( float X, float Y, float Z )
{
  position_[0] = X;
  position_[1] = Y;
  position_[2] = Z;
}
//
//
//
DPo::Point( const DPo& that )
{
  position_[0] = that.position_[0];
  position_[1] = that.position_[1];
  position_[2] = that.position_[2];
}
////
////
////
//DPo::Point( DPo&& that )
//{
//}
//
//
//
DPo::~Point()
{
}
//
//
//
DPo& 
DPo::operator = ( const DPo& that )
{
  position_[0] = that.position_[0];
  position_[1] = that.position_[1];
  position_[2] = that.position_[2];

  //
  //
  return *this;
}
//
//
//
bool
DPo::operator == ( const DPo& that )
{
  return ( position_[0] == that.position_[0] && 
	   position_[1] == that.position_[1] && 
	   position_[2] == that.position_[2] );
}
//
//
//
bool
DPo::operator != ( const DPo& that )
{
  return ( position_[0] != that.position_[0] && 
	   position_[1] != that.position_[1] && 
	   position_[2] != that.position_[2] );
}
////
////
////
//DPo& 
//DPo::operator = ( DPo&& that )
//{
//  if( this != &that )
//    {
////      position_ = that.position_;
////
//    }
//  //
//  return *this;
//}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DPo& that)
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
