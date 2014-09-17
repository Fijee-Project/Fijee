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
/*! \namespace Fijee
 * 
 * Name space for our new package
 *
 */
namespace Fijee
{
  /*! \class Zlib
   *
   * \brief class representing compression process.
   *
   *  This class uses ZLib for compression process.
   */
  class Zlib
  {
  public:
    /*!
     *  \brief 
     *
     *  This methode
     *
     */
    static void in_memory_compression( void*  Data_in, 
				       size_t Data_in_size, 
				       std::vector<Bytef>& Data_out )
    {
      // 
      // Chuncks
      const size_t BUFSIZE = 16384; // 128 * 1024; // = 2^16; 16384 = 2^14
      Bytef temp_buffer[BUFSIZE];
      // Zlib struct
      z_stream strm;
      strm.zalloc    = Z_NULL;
      strm.zfree     = Z_NULL;
      strm.opaque    = Z_NULL;
      strm.next_in   = reinterpret_cast<Bytef*>(Data_in);
      strm.avail_in  = Data_in_size;
      strm.next_out  = temp_buffer;
      strm.avail_out = BUFSIZE;
      // Initialization of the struct to deflate
      deflateInit(&strm, Z_BEST_SPEED);
      // 
      int ret, flush;

      // 
      // Compress until end of file */
      do {
	// Run deflate() on input until output buffer not full, finish
	// compression if all of source has been read in
	flush = (strm.avail_in == 0 ? Z_FINISH : Z_NO_FLUSH);
	do {

	  strm.avail_out = BUFSIZE;
	  strm.next_out = temp_buffer;
	  ret = deflate(&strm, flush);    /* no bad return value */
	  assert(ret != Z_STREAM_ERROR);  /* state not clobbered */

	  Data_out.insert(Data_out.end(), temp_buffer, temp_buffer + BUFSIZE - strm.avail_out);

	} while ( strm.avail_out == 0 );
	// 
	assert( strm.avail_in == 0 );     /* all input will be used */
	// done when last data in file processed
      } while ( flush != Z_FINISH );
      // 
      assert(ret == Z_STREAM_END);        /* stream will be complete */

      // 
      // clean up and return
      (void)deflateEnd(&strm);
    }
    /*!
     *  \brief 
     *
     *  This methode
     *
     */
    static void in_memory_decompression( const std::vector<Bytef>& Data_in, 
					 std::vector<Bytef>& Data_out )
    {
      // 
      // Chuncks
      const size_t BUFSIZE = 16384; // 2^14
      Bytef temp_buffer[BUFSIZE];

      // 
      // Copy the data in a simple strcture
      //  Bytef* data_in = (Bytef*) malloc( Data_in.size() * sizeof(Bytef) );
      Bytef* data_in = new Bytef[Data_in.size()];
      std::copy ( Data_in.begin(), Data_in.end(), data_in );

      // 
      // Zlib struct
      z_stream strm;
      strm.zalloc    = Z_NULL;
      strm.zfree     = Z_NULL;
      strm.opaque    = Z_NULL;
      strm.next_out  = temp_buffer;
      strm.avail_out = BUFSIZE;
      strm.next_in   = Z_NULL;
      strm.avail_in  = 0.;
      // Initialization of the struct to deflate
      inflateInit(&strm);
      //
      strm.next_in   = data_in;
      strm.avail_in  = Data_in.size();
  
      int res = Z_OK;
      //  decompress until deflate stream ends or end of file 
      do {
	// 
	// run inflate() on input until output buffer not full
	do {
	  strm.avail_out = BUFSIZE;
	  strm.next_out = temp_buffer;
	  // 
	  res = inflate(&strm, Z_NO_FLUSH);
	  assert(res != Z_STREAM_ERROR);  /* state not clobbered */
	  switch (res) {
	  case Z_NEED_DICT:
	    res = Z_DATA_ERROR;     /* and fall through */
	  case Z_DATA_ERROR:
	  case Z_MEM_ERROR:
	    (void)inflateEnd(&strm);
	  }
	  // 
	  Data_out.insert(Data_out.end(), temp_buffer, temp_buffer + BUFSIZE - strm.avail_out);
	} while (strm.avail_out == 0);
	// 
	// done when inflate() says it's done
      } while (res != Z_STREAM_END);
  
      // 
      // clean up
      delete[] data_in;
      data_in = nullptr;
      //
      (void)inflateEnd(&strm);
    }
  };
}
#endif
