#include "Electrode.h"
//
// We give a comprehensive type name
//
typedef Domains::Electrode DD;
//
//
//
DD::Electrode():
  Domains::Point_vector(),
  index_(0), label_(""), shape_(nullptr)
{

}
//
//
//
DD::Electrode( int Index, std::string Label, 
	       float Position_x, float Position_y, float Position_z,
	       float Radius, float Phi, float Theta ):
  Domains::Point_vector( Position_x, Position_y, Position_z, 
			 Radius * cos(Phi) * sin(Theta) /*nx*/, 
			 Radius * sin(Phi) * sin(Theta) /*ny*/, 
			 Radius * cos(Theta) /*nz*/, 
			 Radius /*\hat{n} vector normalization*/),
  index_(Index), label_(Label)
{
  shape_.reset( new Sphere() );
// //
// // Check the positions relations
// if( Position_x > Radius * cos(Phi) * sin(Theta) + 0.001 || Position_x < Radius * cos(Phi) * sin(Theta) - 0.001 ||
//     Position_y > Radius * sin(Phi) * sin(Theta) + 0.001 || Position_y < Radius * sin(Phi) * sin(Theta)  - 0.001 ||
//     Position_z > Radius * cos(Theta) + 0.001 || Position_z <  Radius * cos(Theta) - 0.001 )
//   {
//     std::cerr << "Position_x " << Position_x << "\n"
//		<< "Radius * cos(Phi) * sin(Theta) : " << Radius * cos(Phi) * sin(Theta) << "\n"
//		<< "Position_y " << Position_y << "\n"
//		<< "Radius * sin(Phi) * sin(Theta) : " << Radius * sin(Phi) * sin(Theta) << "\n"
//		<< "Position_z " << Position_z << "\n"
//		<< "Radius * cos(Theta): " << Radius * cos(Theta) << "\n"
//		<< "Theta " << Theta << " Phi " << Phi << "\n";
//    exit(1);
//   }
}
//
//
//
DD::Electrode( const DD& that ):
  Domains::Point_vector(that),
  index_(that.index_),label_(that.label_), shape_(that.shape_)
{}
//
//
//
DD::~Electrode(){ /* Do nothing */}
//
//
//
DD& 
DD::operator = ( const DD& that )
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
		       const DD& that)
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
