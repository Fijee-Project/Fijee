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
  intensity_(that.intensity_), potential_(that.potential_),
  impedance_(that.impedance_), shape_(that.shape_)
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
  Domains::Point_vector position_normal = static_cast<Domains::Point_vector> (that);
  position_normal.x() *= 1.e-3;
  position_normal.y() *= 1.e-3;
  position_normal.z() *= 1.e-3;


  //
  //
  stream 
    // Position Direction
    << position_normal
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
