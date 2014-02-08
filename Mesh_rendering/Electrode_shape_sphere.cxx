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
	      float X, float Y, float Z ) const
{
  return ( (Center.x() - X)*(Center.x() - X) + 
	   (Center.y() - Y)*(Center.y() - Y) + 
	   (Center.z() - Z)*(Center.z() - Z) - 
	   radius_ * radius_ < 0. ? true : false );
}
//
//
//
void 
DEss::print( std::ostream& Stream ) const 
{
  Stream 
    // shape type
    << "type=\"SPEHERE\" "
    // shape radius
    << "radius=\"" << radius_ << "\" "
    // shape surface
    << "surface=\"" << contact_surface() << "\" ";
};
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DEss& that)
{
  //
  //
  stream 
    // shape type
    << "type=\"SPEHERE\" "
    // shape radius
    << "radius=\"" << that.get_radius_() << "\" "
    // shape surface
    << "surface=\"" << that.contact_surface() << "\" ";
  
  //
  //
  return stream;
}
