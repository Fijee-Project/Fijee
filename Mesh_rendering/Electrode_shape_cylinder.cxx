#include "Electrode_shape_cylinder.h"
//
// We give a comprehensive type name
//
typedef Domains::Electrode_shape_cylinder DEsc;
//
//
//
DEsc::Electrode_shape_cylinder():
  radius_(0.), height_(0.)
{}
//
//
//
DEsc::Electrode_shape_cylinder( float Radius, float Height ):
  radius_(Radius), height_(Height)
{}
//
//
//
bool
DEsc::inside( Domains::Point_vector& Center, 
	      float X, float Y, float Z ) const
{
  //
  // Gram-Schimdt
  //
  // We are creating an orthogonal base \{ \vec{n}, \vec{n_tild} \}
  // Then we are projecting \vec{CP} ont this base. The projection must 
  // be in the rectangle (2 x radius_ x height).
  Eigen::Vector3f CP;
  CP <<
    X - Center.x(),
    Y - Center.y(),
    Z - Center.z();
  //
  Eigen::Vector3f n;
  n <<
    Center.vx(),
    Center.vy(),
    Center.vz();
  // Project of \vec{CP} on \vec{n}
  float proj_n = CP.dot(n);

  //
  // \vec{n_tild} orthogonal to \vec{n}
  Eigen::Vector3f n_tild = CP - proj_n * n;
  n_tild.normalize();
  //
  float proj_n_tild = CP.dot(n_tild);


  //
  //
  return ( ( abs( proj_n ) < height_ / 2. && abs( proj_n_tild ) < radius_ ) ? 
	   true : false );
}
//
//
//
void 
DEsc::print( std::ostream& Stream ) const 
{
  Stream 
    // shape type
    << "type=\"CYLINDER\" "
    // shape radius
    << "radius=\"" << radius_ << "\" "
    // shape height
    << "height=\"" << height_ << "\" "
    // shape surface
    << "surface=\"" << contact_surface() << "\" ";
};
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DEsc& that)
{
  //
  //
  stream 
    // shape type
    << "type=\"CYLINDER\" "
    // shape radius
    << "radius=\"" << that.get_radius_() << "\" "
    // shape height
    << "height=\"" << that.get_height_() << "\" "
    // shape surface
    << "surface=\"" << that.contact_surface() << "\" ";
  
  //
  //
  return stream;
}
