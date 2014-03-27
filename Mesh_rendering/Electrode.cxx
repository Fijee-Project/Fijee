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
  index_(0), label_(""), intensity_(0.), 
  potential_(0.), impedance_(std::complex<float>(0.,0.)),
  shape_(nullptr)
{}
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
  impedance_ = std::complex<float>(0.,0.);
  intensity_ = 0. /* A */;
  potential_ = 0. /* V */;
  
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
	float height = 3 /*mm*/;
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
  index_(that.index_),label_(that.label_), 
  intensity_(that.intensity_), impedance_(that.impedance_),
  potential_(that.potential_), shape_(that.shape_)
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
  index_     = that.index_;
  label_     = that.label_;
  intensity_ = that.intensity_;
  potential_ = that.potential_;
  impedance_ = that.impedance_;
  shape_     = that.shape_;

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
    // Electrode label
    << "label=\"" << that.get_label_() << "\" "
    // Electrode intensity
    << "I=\"" << that.get_intensity_() << "\" "
    // Electrode potential
    << "V=\"" << that.get_potential_() << "\" "
    // Electrode impedence 
    << "Re_z_l=\"" << that.get_impedance_().real() << "\" "
    << "Im_z_l=\"" << that.get_impedance_().imag() << "\" ";
    // Electrode shape
    that.get_shape_()->print( stream );
  
  //
  //
  return stream;
}
