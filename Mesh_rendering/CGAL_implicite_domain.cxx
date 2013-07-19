//
// Project
//
#include "CGAL_implicite_domain.h"
//
// We give a comprehensive type name
//
typedef Domains::CGAL_implicite_domain Domain;
//
//
//
Domain::CGAL_implicite_domain():
  binary_mask_(""),
  select_enclosed_points_(nullptr)
{
}
//
//
//
Domain::CGAL_implicite_domain( const char* Binary_Mask ):
  binary_mask_( Binary_Mask ),
  select_enclosed_points_(nullptr)
{}
//
//
//
void
Domain::operator ()( float** Space_Points )
{
  //
  //
  Gray_level_image image(binary_mask_.c_str(), .0f);

  //
  // Carefully choosen bounding sphere: the center must be inside the
  // surface defined by 'image' and the radius must be high enough so that
  // the sphere actually bounds the whole image.
  GT::Point_3  bounding_sphere_center(256./2., 256./2., 80.0);
  GT::FT       bounding_sphere_squared_radius = 200.*2.;
  GT::Sphere_3 bounding_sphere( bounding_sphere_center,
			        bounding_sphere_squared_radius);
  
  //
  //
  select_enclosed_points_ = new CGAL::Implicit_surface_3<GT, Gray_level_image> ( image, 
										 bounding_sphere, 
										 1e-2);
}
//
//
//
Domain::CGAL_implicite_domain( const Domain& that )
{
}
//
//
//
Domain::CGAL_implicite_domain( Domain&& that )
{
}
//
//
//
Domain::~CGAL_implicite_domain()
{
  if( select_enclosed_points_ )
    {
      delete select_enclosed_points_;
      select_enclosed_points_ = nullptr;
    }
}
//
//
//
Domain& 
Domain::operator = ( const Domain& that )
{
  if ( this != &that ) 
    {
//      // free existing ressources
//      if( tab_ )
//	{
//	  delete [] tab_;
//	  tab_ = nullptr;
//	}
//      // allocating new ressources
//      pos_x_ = that.get_pos_x();
//      pos_y_ = that.get_pos_y();
//      list_position_ = that.get_list_position();
//      //
//      tab_ = new int[4];
//      std::copy( &that.get_tab(),  &that.get_tab() + 4, tab_ );
    }
  //
  return *this;
}
//
//
//
Domain& 
Domain::operator = ( Domain&& that )
{
  if( this != &that )
    {
//      // initialisation
//      pos_x_ = 0;
//      pos_y_ = 0;
//      delete [] tab_;
//      tab_   = nullptr;
//      // pilfer the source
//      list_position_ = std::move( that.list_position_ );
//      pos_x_ =  that.get_pos_x();
//      pos_y_ =  that.get_pos_y();
//      tab_   = &that.get_tab();
//      // reset that
//      that.set_pos_x( 0 );
//      that.set_pos_y( 0 );
//      that.set_tab( nullptr );
    }
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const Domain& that)
{
//  std::for_each( that.get_list_position().begin(),
//		 that.get_list_position().end(),
//		 [&stream]( int Val )
//		 {
//		   stream << "list pos = " << Val << "\n";
//		 });
//  //
//  stream << "positions minimum = " 
//	 << that.get_min_x() << " "
//	 << that.get_min_y() << " "
//	 << that.get_min_z() << "\n";
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
