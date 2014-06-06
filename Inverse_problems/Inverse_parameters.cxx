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
#include "Inverse_parameters.h"
//
// We give a comprehensive type name
//
typedef Inverse::Inverse_parameters IIp;//SDEsp;
typedef struct stat stat_file;
//
//
//
IIp*
IIp::parameters_instance_ = NULL;
//
//
//
IIp::Inverse_parameters()
{
  //
  // Check on ENV variables
  Utils::Fijee_environment fijee;
  //
  files_path_         = fijee.get_fem_path_();
  files_path_output_  = fijee.get_fem_output_path_();
  files_path_result_  = fijee.get_fem_result_path_();
  files_path_measure_ = fijee.get_fem_measure_path_();

  // 
  // Time profiler lof file
  // It the file existes: empty it.
#ifdef TIME_PROFILER
  std::ofstream ofs ( "fijee_time.log", std::ofstream::app );
  if( ofs.good() ) ofs.clear();
  ofs.close();
#endif
}
//
//
//
IIp::Inverse_parameters( const IIp& that ){}
//
//
//
IIp::~Inverse_parameters()
{}
//
//
//
IIp& 
IIp::operator = ( const IIp& that )
{
  //
  return *this;
}
//
//
//
IIp* 
IIp::get_instance()
{
  if( parameters_instance_ == NULL )
    parameters_instance_ = new IIp();
  //
  return parameters_instance_;
}
//
//
//
void 
IIp::kill_instance()
{
  if( parameters_instance_ != NULL )
    {
      delete parameters_instance_;
      parameters_instance_ = NULL;
    }
}
//
//
//
void 
IIp::init()
{

  //
  // Dispatching information
  number_of_threads_ = 2;
}
//
//
//
std::ostream& 
Inverse::operator << ( std::ostream& stream, 
		      const IIp& that)
{
  stream << " Pattern Singleton\n";
  //
  return stream;
}
