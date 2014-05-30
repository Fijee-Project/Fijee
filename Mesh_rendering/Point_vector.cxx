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
#include "Point_vector.h"
//
// We give a comprehensive type name
//
typedef Domains::Point_vector DPv;
//
//
//
DPv::Point_vector(): Domains::Point()
{
  vector_[0] = 0.;
  vector_[1] = 0.;
  vector_[2] = 0.;
  //
  norm_ = 1.;
}
//
//
//
DPv::Point_vector(  float X, float Y, float Z, 
		    float Vx, float Vy, float Vz,
		    float Weight ): 
  Domains::Point(X,Y,Z,Weight)
{
  vector_[0] = Vx;
  vector_[1] = Vy;
  vector_[2] = Vz;
  //
  normalize();
}
//
//
//
DPv::Point_vector( const DPv& that ): 
  Domains::Point(that),
  norm_(that.norm_)
{
  vector_[0] = that.vector_[0];
  vector_[1] = that.vector_[1];
  vector_[2] = that.vector_[2];
}
//
//
//
DPv::~Point_vector()
{
}
//
//
//
DPv& 
DPv::operator = ( const DPv& that )
{
  Domains::Point::operator = (that);
  //
  vector_[0] = that.vector_[0];
  vector_[1] = that.vector_[1];
  vector_[2] = that.vector_[2];
  //
  norm_ = that.norm_;
 
  //
  //
  return *this;
}
//
//
//
bool
DPv::operator == ( const DPv& that )
{
  return ( Domains::Point::operator == (that) && 
	   vector_[0] == that.vector_[0] && 
	   vector_[1] == that.vector_[1] && 
	   vector_[2] == that.vector_[2] );
}
//
//
//
bool
DPv::operator != ( const DPv& that )
{
  return ( Domains::Point::operator != (that) && 
	   vector_[0] != that.vector_[0] && 
	   vector_[1] != that.vector_[1] && 
	   vector_[2] != that.vector_[2] );
}
//
//
//
DPv& 
DPv::operator + ( const DPv& Vector)
{
  DPv* vector = new DPv(*this);

  //
  //
  vector->vx() += Vector.vx();
  vector->vy() += Vector.vy();
  vector->vz() += Vector.vz();
  //
  normalize();

  //
  //
  return *vector;
}
//
//
//
DPv& 
DPv::operator += ( const DPv& Vector)
{
  //
  //
  vector_[0] += Vector.vx();
  vector_[1] += Vector.vy();
  vector_[2] += Vector.vz();
  //
  normalize();

  //
  //
  return *this;
}
//
//
//
DPv& 
DPv::operator - ( const DPv& Vector)
{
  DPv* vector = new DPv(*this);

  //
  //
  vector->vx() -= Vector.vx();
  vector->vy() -= Vector.vy();
  vector->vz() -= Vector.vz();
  //
  normalize();

  //
  //
  return *vector;
}
//
//
//
DPv& 
DPv::operator -= ( const DPv& Vector)
{
  //
  //
  vector_[0] -= Vector.vx();
  vector_[1] -= Vector.vy();
  vector_[2] -= Vector.vz();
  //
  normalize();

  //
  //
  return *this;
}
//
//
//
DPv& 
DPv::cross( const DPv& Vector )
{
  DPv* vector = new DPv();
  //
  vector->vx() = vector_[1] * Vector.vz() - vector_[2] * Vector.vy();
  vector->vy() = vector_[2] * Vector.vx() - vector_[0] * Vector.vz();
  vector->vz() = vector_[0] * Vector.vy() - vector_[1] * Vector.vx();
  //
  vector->normalize();

  //
  //
  return *vector;
}
//
//
//
float 
DPv::dot( const Point_vector& Vector ) const
{
  float scalar = 0.;
  //
  for( int coord = 0 ; coord < 3 ; coord++ )
    scalar += vector_[coord] * ( Vector.get_vector_() )[coord];

  //
  //
  return scalar;
}
//
//
//
float 
DPv::cosine_theta( const Point_vector& Vector ) const
{
  float cosine_theta = 0.;
  //
  if( norm_ == 0 || Vector.get_norm_() == 0 )
    {
      std::cerr << "Vector norm is null." << std::endl;
    }
  //
  cosine_theta = dot( Vector );
  cosine_theta /= norm_ * Vector.get_norm_();
  //
  //
  return cosine_theta;
}
//
//
//
void
DPv::normalize()
{
  norm_ = sqrt( dot(*this) );
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DPv& that)
{
  //
  //
  stream << static_cast<Domains::Point>(that)
	 << "vx=\"" << that.vx() << "\" vy=\"" << that.vy() << "\" vz=\"" << that.vz() << "\" ";
  
  //
  //
  return stream;
};
