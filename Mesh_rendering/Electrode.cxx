#include "Electrode.h"
//
// We give a comprehensive type name
//
typedef Domains::Electrode DD;
//
//
//
DD::Electrode():
  Domains::Point_vector()
{

}
//
//
//
DD::Electrode( int index, std::string label, 
	       float position_x, float position_y, float position_z,
	       float radius, float phi, float theta )
{
  Domains::Point_vector( position_x, position_y, position_z, 
			 radius * cos(theta) * sin(phi) /*nx*/, 
			 radius * sin(theta) * sin(phi) /*ny*/, 
			 radius * cos(phi) /*nz*/, 
			 radius /*\hat{n} vector normalization*/);
}
//
//
//
DD::Electrode( const DD& that ):
  Domains::Point_vector(that)
{

}
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
// stream 
//    // Position Direction
//    << static_cast<Domains::Point_vector> (that)
//    // Electrode intensity
//    << "I=\"" << that.weight() << "\" "
////    // Conductivity coefficients
////    << "C00=\"" << that.C00() << "\" C01=\"" << that.C01() << "\" C02=\"" << that.C02() << "\" "
////    << "C11=\"" << that.C11() << "\" C12=\"" << that.C12() << "\" C22=\"" << that.C22() << "\" ";
//    // Conductivity eigenvalues
//    << "lambda1=\"" << that.lambda1() 
//    << "\" lambda2=\"" << that.lambda2() 
//    << "\" lambda3=\"" << that.lambda3() << "\" ";
  
  //
  //
  return stream;
}
