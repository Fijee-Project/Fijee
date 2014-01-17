#include "Electrode.h"
//
// We give a comprehensive type name
//
typedef Domains::Electrode DE;
//
//
//
DE::Electrode():
  Domains::Point_vector(),
  index_(0), label_(""), shape_(nullptr)
{

}
//
//
//
DE::Electrode( int Index, std::string Label, 
	       float Position_x, float Position_y, float Position_z,
	       float Radius, float Phi, float Theta ):
  Domains::Point_vector( Position_x, Position_y, Position_z, 
			 Radius * cos(Phi) * sin(Theta) /*nx*/, 
			 Radius * sin(Phi) * sin(Theta) /*ny*/, 
			 Radius * cos(Theta) /*nz*/, 
			 Radius /*\hat{n} vector normalization*/),
  index_(Index), label_(Label)
{
  //
  // Select the shape factory
  // ToDo Parameter from Access_parameters
  Electrode_type Shape = CYLINDER;
  
  switch ( Shape )
    {
    case SPHERE:
      {
	// ToDo Parameter from Access_parameters
	float radius = 5 /*mm*/;
	shape_.reset( new Electrode_shape_sphere(radius) );
	break;
      }
    case CYLINDER:
      {
	// ToDo Parameter from Access_parameters
	float radius = 5 /*mm*/;
	float height = 2 /*mm*/;
	shape_.reset( new Electrode_shape_cylinder(radius, height) );
	break;
      }
    default:
      {
	std::cerr << "Electrode shape unknown" << std::endl;
	abort();
      }
    }
}
//
//
//
DE::Electrode( const DE& that ):
  Domains::Point_vector(that),
  index_(that.index_),label_(that.label_), shape_(that.shape_)
{}
//
//
//
DE::~Electrode(){ /* Do nothing */}
//
//
//
DE& 
DE::operator = ( const DE& that )
{
  Domains::Point_vector::operator = (that);
  //
  index_ = that.index_;
  label_ = that.label_;
  shape_ = that.shape_;

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DE& that)
{
  //
  //
  stream 
    // Position Direction
    << static_cast<Domains::Point_vector> (that)
    // Dipole intensity
    << "label=\"" << that.get_label_() << "\" ";
  
  //
  //
  return stream;
}
