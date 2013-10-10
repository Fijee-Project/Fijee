#include "Point.h"
//
// We give a comprehensive type name
//
typedef Domains::Point DPo;
//
//
//
DPo::Point():
  weight_(1.)
{
  position_[0] = position_[1] = position_[2] = 0.;
}
//
//
//
DPo::Point( float X, float Y, float Z, float Weight ):
  weight_(Weight)
{
  position_[0] = X;
  position_[1] = Y;
  position_[2] = Z;
}
//
//
//
DPo::Point( const DPo& that ):
  weight_(that.weight_)
{
  position_[0] = that.position_[0];
  position_[1] = that.position_[1];
  position_[2] = that.position_[2];
}
//
//
//
DPo::~Point()
{
}
//
//
//
DPo& 
DPo::operator = ( const DPo& that )
{
  weight_ = that.weight_;
  //
  position_[0] = that.position_[0];
  position_[1] = that.position_[1];
  position_[2] = that.position_[2];

  //
  //
  return *this;
}
//
//
//
bool
DPo::operator == ( const DPo& that )
{
  return ( weight_      == that.weight_ &&
	   position_[0] == that.position_[0] && 
	   position_[1] == that.position_[1] && 
	   position_[2] == that.position_[2] );
}
//
//
//
bool
DPo::operator != ( const DPo& that )
{
  return ( weight_      != that.weight_ &&
	   position_[0] != that.position_[0] && 
	   position_[1] != that.position_[1] && 
	   position_[2] != that.position_[2] );
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DPo& that)
{
  //
  //
  stream << "x=\"" << that.x() << "\" y=\"" << that.y() << "\" z=\"" << that.z() << "\" ";

  //
  //
  return stream;
};
