#include "Build_dipoles_list.h"
//
// We give a comprehensive type name
//
typedef Domains::Build_dipoles_list DBdl;
//
//
//
DBdl::Build_dipoles_list()
{
}
//
//
//
DBdl::Build_dipoles_list( const DBdl& that )
{
}
//
//
//
DBdl::~Build_dipoles_list()
{
  /* Do nothing */
}
//
//
//
DBdl& 
DBdl::operator = ( const DBdl& that )
{

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DBdl& that)
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
