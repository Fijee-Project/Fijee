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
#include "Electrode_shape_sphere.h"
//
// We give a comprehensive type name
//
typedef Domains::Electrode_shape_sphere DEss;
//
//
//
DEss::Electrode_shape_sphere():
  radius_(0.)
{
}
//
//
//
DEss::Electrode_shape_sphere( float Radius ):
  radius_(Radius)
{
}
//
//
//
bool
DEss::inside( Domains::Point_vector& Center, 
	      float X, float Y, float Z ) const
{
  return ( (Center.x() - X)*(Center.x() - X) + 
	   (Center.y() - Y)*(Center.y() - Y) + 
	   (Center.z() - Z)*(Center.z() - Z) - 
	   radius_ * radius_ < 0. ? true : false );
}
//
//
//
void 
DEss::print( std::ostream& Stream ) const 
{
  Stream 
    // shape type
    << "type=\"SPEHERE\" "
    // shape radius; [mm] to [m]
    << "radius=\"" << radius_ * 1.e-3 << "\" "
    // shape surface; [mm] to [m]
    << "surface=\"" << contact_surface() * 1.e-6 << "\" ";
};
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DEss& that)
{
  //
  //
  stream 
    // shape type
    << "type=\"SPEHERE\" "
    // shape radius; [mm] to [m]
    << "radius=\"" << that.get_radius_() * 1.e-3 << "\" "
    // shape surface; [mm] to [m]
    << "surface=\"" << that.contact_surface() * 1.e-6 << "\" ";
  
  //
  //
  return stream;
}
