//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
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
