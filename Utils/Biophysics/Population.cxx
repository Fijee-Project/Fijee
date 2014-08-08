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
#include "Population.h"
//
// We give a comprehensive type name
//
typedef Utils::Biophysics::Population UBPo;
//
//
//
UBPo::Population():
  I_(0.), index_(0), index_cell_(0), index_parcel_(0)
{
  position_[0]  = position_[1]  = position_[2]  = 0.;
  direction_[0] = direction_[1] = direction_[2] = 0.;
  lambda_[0]    = lambda_[1]    = lambda_[2]    = 0.;
}
//
//
//
UBPo::Population( int Index, 
		  double X, double Y, double Z, 
		  double Vx, double Vy, double Vz,
		  double I, int Index_cell, int Index_parcel,
		  double Lambda1, double Lambda2, double Lambda3 ):
  I_(I), index_(Index), index_cell_(Index_cell), index_parcel_(Index_parcel)
{
  position_[0] = X;
  position_[1] = Y;
  position_[2] = Z;
  //
  direction_[0] = Vx;
  direction_[1] = Vy;
  direction_[2] = Vz;
  //
  lambda_[0] = Lambda1;
  lambda_[1] = Lambda2;
  lambda_[2] = Lambda3;
}
//
//
//
UBPo::Population( const UBPo& that ):
  I_(that.I_), 
  index_(that.index_), index_cell_(that.index_cell_), index_parcel_(that.index_parcel_)
{
  position_[0] = that.position_[0];
  position_[1] = that.position_[1];
  position_[2] = that.position_[2];
  //
  direction_[0] = that.direction_[0];
  direction_[1] = that.direction_[1];
  direction_[2] = that.direction_[2];
  //
  lambda_[0] = that.lambda_[0];
  lambda_[1] = that.lambda_[1];
  lambda_[2] = that.lambda_[2];
  // 
  V_time_series_ = that.V_time_series_;
}
//
//
//
UBPo::Population( UBPo&& that ):
  I_(0.), index_(0), index_cell_(0), index_parcel_(0)

{
  // 
  // Resources initialization
  position_[0]  = position_[1]  = position_[2]  = 0.;
  direction_[0] = direction_[1] = direction_[2] = 0.;
  lambda_[0]    = lambda_[1]    = lambda_[2]    = 0.;
  // 
  V_time_series_.clear();
  

  // 
  // Resources attribution from that object
  position_[0] = that.position_[0];
  position_[1] = that.position_[1];
  position_[2] = that.position_[2];
  //
  direction_[0] = that.direction_[0];
  direction_[1] = that.direction_[1];
  direction_[2] = that.direction_[2];
  //
  lambda_[0] = that.lambda_[0];
  lambda_[1] = that.lambda_[1];
  lambda_[2] = that.lambda_[2];
  // 
  I_ = that.I_;
  index_        = that.index_;
  index_cell_   = that.index_cell_;
  index_parcel_ = that.index_parcel_;
  // 
  V_time_series_ = std::move(that.V_time_series_);

  // 
  // That object ressources initialization
  that.position_[0] = 0.;
  that.position_[1] = 0.;
  that.position_[2] = 0.;
  //
  that.direction_[0] = 0.;
  that.direction_[1] = 0.;
  that.direction_[2] = 0.;
  //
  that.lambda_[0] = 0.;
  that.lambda_[1] = 0.;
  that.lambda_[2] = 0.;
  // 
  that.I_ = 0.;
  that.index_        = 0;
  that.index_cell_   = 0;
  that.index_parcel_ = 0;
  // 
  that.V_time_series_.clear();
}
//
//
//
UBPo& 
UBPo::operator = ( const UBPo& that )
{
  if ( this != &that ) 
    {
      position_[0] = that.position_[0];
      position_[1] = that.position_[1];
      position_[2] = that.position_[2];
      //
      direction_[0] = that.direction_[0];
      direction_[1] = that.direction_[1];
      direction_[2] = that.direction_[2];
      //
      lambda_[0] = that.lambda_[0];
      lambda_[1] = that.lambda_[1];
      lambda_[2] = that.lambda_[2];
      // 
      I_ = that.I_;
      index_        = that.index_;
      index_cell_   = that.index_cell_;
      index_parcel_ = that.index_parcel_;
      // 
      V_time_series_ = that.V_time_series_;
    }
 
  //
  //
  return *this;
}
//
//
//
UBPo& 
UBPo::operator = ( UBPo&& that )
{
  if ( this != &that ) 
    {
      // 
      // Resources initialization
      position_[0]  = position_[1]  = position_[2]  = 0.;
      direction_[0] = direction_[1] = direction_[2] = 0.;
      lambda_[0]    = lambda_[1]    = lambda_[2]    = 0.;
      // 
      I_ = 0.;
      index_        = 0;
      index_cell_   = 0;
      index_parcel_ = 0;
      // 
      V_time_series_.clear();
     
      // 
      // Resources attribution from that object
      position_[0] = that.position_[0];
      position_[1] = that.position_[1];
      position_[2] = that.position_[2];
      //
      direction_[0] = that.direction_[0];
      direction_[1] = that.direction_[1];
      direction_[2] = that.direction_[2];
      //
      lambda_[0] = that.lambda_[0];
      lambda_[1] = that.lambda_[1];
      lambda_[2] = that.lambda_[2];
      // 
      I_ = that.I_;
      index_        = that.index_;
      index_cell_   = that.index_cell_;
      index_parcel_ = that.index_parcel_;
      // 
      V_time_series_ = std::move(that.V_time_series_);
     
      // 
      // That object ressources initialization
      that.position_[0] = 0.;
      that.position_[1] = 0.;
      that.position_[2] = 0.;
      //
      that.direction_[0] = 0.;
      that.direction_[1] = 0.;
      that.direction_[2] = 0.;
      //
      that.lambda_[0] = 0.;
      that.lambda_[1] = 0.;
      that.lambda_[2] = 0.;
      // 
      that.I_ = 0.;
      that.index_        = 0;
      that.index_cell_   = 0;
      that.index_parcel_ = 0;
      // 
      that.V_time_series_.clear();
    }
  
  //
  //
  return *this;
}
//
//
//
std::ostream& 
Utils::Biophysics::operator << ( std::ostream& stream, 
				 const UBPo& that)
{
  //
  //
  //  stream << "x=\"" << that.x() << "\" y=\"" << that.y() << "\" z=\"" << that.z() << "\" ";

  //
  //
  return stream;
}
