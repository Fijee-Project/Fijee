#include <stdio.h>
#include <cstdio>
#include <sstream>
#include <algorithm> // std::for_each()
//
// UCSF
//
#include "Spheres_conductivity_tensor.h"
#include "Access_parameters.h"
//
// VTK
//
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>
// Transformation
#include <vtkTransform.h>
// Geometry
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkArrowSource.h>
#include <vtkSphereSource.h>
#include <vtkCubeSource.h>
#include <vtkConeSource.h>
#include <vtkAxesActor.h>
#include <vtkGlyph3D.h>
// Rendering
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
//#include <vtkProgrammableGlyphFilter.h>
// Include the mesh data
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkDataSetMapper.h>
//
// We give a comprehensive type name
//
typedef Domains::Spheres_conductivity_tensor DSct;
typedef Domains::Access_parameters DAp;
//
//
//
DSct::Spheres_conductivity_tensor()
{}
//
//
//
DSct::Spheres_conductivity_tensor( const DSct& that )
{
//  std::copy( &that.get_tab(),  &that.get_tab() + 4, tab_ );
////  tab_[0] = ( &that.get_tab() )[0];
////  tab_[1] = ( &that.get_tab() )[1];
////  tab_[2] = ( &that.get_tab() )[2];
////  tab_[3] = ( &that.get_tab() )[3];
}
//
//
//
DSct::Spheres_conductivity_tensor( DSct&& that )
{
//  // pilfer the source
//  list_position_ = std::move( that.list_position_ );
//  pos_x_ =  that.get_pos_x();
//  pos_y_ =  that.get_pos_y();
//  tab_   = &that.get_tab();
//  // reset that
//  that.set_pos_x( 0 );
//  that.set_pos_y( 0 );
//  that.set_tab( nullptr );
}
//
//
//
DSct& 
DSct::operator = ( const DSct& that )
{
//  if ( this != &that ) 
//    {
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
//    }
//  //
  return *this;
}
//
//
//
DSct& 
DSct::operator = ( DSct&& that )
{
//  if( this != &that )
//    {
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
//    }
  //
  return *this;
}
//
//
//
void
DSct::operator ()()
{
}
//
//
//
void 
DSct::move_conductivity_array_to_parameters()
{}
//
//
//
void 
DSct::VTK_visualization()
{}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DSct& that)
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
