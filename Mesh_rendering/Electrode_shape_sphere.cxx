#include "Electrode_shape_sphere.h"
//
// We give a comprehensive type name
//
typedef Domains::Electrode_shape_sphere DEss;
//
//
//
DEss::Electrode_shape_sphere():
  radius_(0.)
{
}
//
//
//
DEss::Electrode_shape_sphere( float Radius ):
  radius_(Radius)
{
}
//
//
//
bool
DEss::inside( Domains::Point_vector& Center, 
	      float X, float Y, float Z )
{
  return ( (Center.x() - X)*(Center.x() - X) + 
	   (Center.y() - Y)*(Center.y() - Y) + 
	   (Center.z() - Z)*(Center.z() - Z) - 
	   radius_ * radius_ < 0. ? true : false );
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DEss& that)
{
  //
  //
//  stream 
//    // Position Direction
//    << static_cast<Domains::Point_vector> (that)
//    // Dipole intensity
//    << "label=\"" << that.get_label_() << "\" ";
  
  //
  //
  return stream;
}
