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
#include "Leadfield_matrix.h"
//
// We give a comprehensive type name
//
typedef Utils::Biophysics::Leadfield_matrix UBLm;
//
//
//
UBLm::Leadfield_matrix():
  index_(0),label_("")
{
}
//
//
//
UBLm::Leadfield_matrix( const int Index, const std::string Label):
  index_(Index),label_(Label)
{
}
//
//
//
UBLm::Leadfield_matrix( const UBLm& that ):
  index_(that.index_), label_(that.label_)
{
  V_dipole_ = that.V_dipole_;
}
//
//
//
UBLm::Leadfield_matrix( UBLm&& that ):
  index_(0), label_(std::string(""))

{
  // 
  // Resources initialization
  V_dipole_.clear();
  

  // 
  // Resources attribution from that object
  index_ = that.index_;
  label_ = that.label_;
  // 
  V_dipole_ = std::move(that.V_dipole_);

  // 
  // That object ressources initialization
  that.index_ = 0;
  that.label_ = std::string("");
  // 
  that.V_dipole_.clear();
}
//
//
//
UBLm& 
UBLm::operator = ( const UBLm& that )
{
  if ( this != &that ) 
    {
      index_ = that.index_;
      label_ = that.label_;
      // 
      V_dipole_ = that.V_dipole_;
    }
 
  //
  //
  return *this;
}
//
//
//
UBLm& 
UBLm::operator = ( UBLm&& that )
{
  if ( this != &that ) 
    {
      // 
      // Resources initialization
      index_ = 0;
      label_ = std::string("0");
      // 
      V_dipole_.clear();
      
      // 
      // Resources attribution from that object
      index_ = that.index_;
      label_ = that.label_;
      // 
      V_dipole_ = std::move(that.V_dipole_);
      
      // 
      // That object ressources initialization
      that.index_ = 0;
      that.label_ = std::string("");
      // 
      that.V_dipole_.clear();
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
				 const UBLm& that)
{
  //
  //
  stream << "index=\"" << that.get_index_() << "\" label=\"" << that.get_label_() << "\" \n";
  for( auto v : that.get_V_dipole_() )
    stream << v << " ";
  stream << std::endl;

  //
  //
  return stream;
}
