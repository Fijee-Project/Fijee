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
#ifndef FIJEE_TURNAROUND_H
#define FIJEE_TURNAROUND_H
/*!
 * \file Fijee_turnaround.h
 * \brief Log management 
 * \author Yann Cobigo
 * \version 0.1
 */
//
//
//
/*! \namespace Fijee
 * 
 * Name space for our new package
 *
 */
namespace Fijee
{
  /*
   * This source is a turnaround to access private members from an other class from a library mostlikely unreachable.
   * The turnaround is due to:
   * http://bloglitb.blogspot.com/2010/07/access-to-private-members-thats-easy.html
   *
   */
  template<typename Tag>
    struct result {
      /* export it ... */
      typedef typename Tag::type type;
      static type ptr;
    };

  template<typename Tag>
    typename result<Tag>::type result<Tag>::ptr;
  
  template<typename Tag, typename Tag::type p>
    struct rob : result<Tag> {
    /* fill it ... */
    struct filler {
      filler() { result<Tag>::ptr = p; }
    };
    static filler filler_obj;
  };
  
  template<typename Tag, typename Tag::type p>
    typename rob<Tag, p>::filler rob<Tag, p>::filler_obj;
}
#endif
