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
#ifndef BRAIN_RHYTHM_H
#define BRAIN_RHYTHM_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Brain_rhythm.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
// UCSF
//
#include "Utils/Statistical_analysis.h"
// 
// 
// 
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \namespace Minimizers
   * 
   * Name space for our new package
   *
   */
  namespace Biophysics
  {
    /*! \class Brain_rhythm
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class Brain_rhythm: public Utils::Statistical_analysis

    {
    public:
      virtual ~Brain_rhythm(){/* Do nothing */};  
      
    public:
      //      virtual void initialization() = 0;
      virtual void modelization() = 0;
    };
  }
}
#endif
