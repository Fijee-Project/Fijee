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
#include "../../Point_vector.h"

int
main()
{
  //
  //
  std::cout << "Algebra" << std::endl;
  Domains::Point_vector V1(1.,2.,3.,10.,0.,0.);
  Domains::Point_vector V2(1.,2.,3.,5.,2.,0.);
  //
  std::cout << V1.get_norm_() << std::endl;
  std::cout << V2.get_norm_() << std::endl;
  std::cout << V1.dot(V2) << std::endl;
  std::cout << V1.cosine_theta(V2) << std::endl;
  std::cout << V1.cross(V2).vx() << " "  
	    << V1.cross(V2).vy() << " "   
	    << V1.cross(V2).vz() 
	    << std::endl;

  //
  //
  std::cout << "Somme" << std::endl;
  Domains::Point_vector P1(0.,2.,4.,1.,0.,0.);
  Domains::Point_vector P2(7.,3.,4.,0.,2.,0.);
  Domains::Point_vector P3(7.,8.,3.,0.,0.,3.);
  Domains::Point_vector P4(1.,5.,9.,4.,0.,0.);
  //
  std::cout << (P1).get_norm_() << std::endl;
  std::cout << (P1 + P2).vx() << " " << (P1 + P2).vy() << " " << (P1 + P2).vz()
	    << std::endl;
  std::cout << (P1 += P2).get_norm_() << std::endl;
  std::cout << (P1).get_norm_()
	    << std::endl;
  std::cout << (P3 - P4).vx() << " " << (P3 - P4).vy() << " " << (P3 - P4).vz()
	    << std::endl;





  //
  //
  return EXIT_SUCCESS;
}
