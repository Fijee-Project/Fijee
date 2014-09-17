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
#ifndef FIJEE_COMPRESSION_H
#define FIJEE_COMPRESSION_H
//
//
//
#include <zlib.h>
#include <vector>
#include <cassert>
#include <algorithm>
//
//
//
/*!
 * \file Fijee_compression.h
 * \brief Compression functions
 * \author Yann Cobigo
 * \version 0.1
 */
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
  namespace Zlib
  {
    /*! \class Compression
     * \brief classe representing compression preocss.
     *
     *  This class is 
     */
    class Compression
    {
    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Compression
       *
       */
    Compression(){};
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
    Compression( const Compression& that) = default;
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Compression
       */
      virtual ~Compression(){/*Do nothing*/};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Compression
       *
       */
      Compression& operator = ( const Compression& that) = default;


    public:
      /*!
       *  \brief 
       *
       *  This methode
       *
       */
      void in_memory_compression( void*, size_t, std::vector<Bytef>& );
      /*!
       *  \brief 
       *
       *  This methode
       *
       */
      void in_memory_decompression(const std::vector<Bytef>&, std::vector<Bytef>&);
    };
  }
}
#endif
