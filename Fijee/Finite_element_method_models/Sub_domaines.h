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
#ifndef _SUB_DOMAINES_H
#define _SUB_DOMAINES_H
#include <dolfin.h>

using namespace dolfin;

//
// Sub domain for Dirichlet boundary condition
// r = 78
class Brain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.077 + 0.001 );
  }
};

// r = 80
class CSF : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) > 0.077  && 
	     sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.080 + 0.002 );
  }
};

// r = 86
class Skull : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1)  ) > 0.080 + DOLFIN_EPS && 
	     sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.086  );
  }
};

// r = 92
class Scalp : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ( sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) > 0.086  && 
	     sqrt( (x[0] - 0.1)*(x[0] - 0.1) + 
		   (x[1] - 0.1)*(x[1] - 0.1) +
		   (x[2] - 0.1)*(x[2] - 0.1) ) <= 0.092 + DOLFIN_EPS || on_boundary );
  }
};

#endif
