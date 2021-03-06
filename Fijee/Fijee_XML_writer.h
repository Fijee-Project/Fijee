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
#ifndef FIJEE_XML_WRITER_H
#define FIJEE_XML_WRITER_H
/*!
 * \file Fijee_XML_writer.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <sstream>
#include "Utils/Third_party/pugi/pugixml.hpp"
/*! \namespace Fijee
 * 
 * Name space for our new package
 *
 */
namespace Fijee
{
  /*! \class XML_writer
   * \brief class representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class XML_writer
  {
  private:
    //! Memory managment member. This member is paired with out_XML_file_ in heavy load managment.
    bool memory_heavy_load_;

  protected:
    //! XML file name
    std::string file_name_;
   //! XML document
    pugi::xml_document document_;
    //! Root of the XML tree
    pugi::xml_node fijee_;

    
 public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class XML_writer
     *
     */
  XML_writer():memory_heavy_load_(false){};
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class XML_writer
     *
     */
  XML_writer( std::string File_name, bool Memory_heavy_load = false ):
    memory_heavy_load_(Memory_heavy_load), file_name_( File_name )
      {
 	// Main node fijee
	fijee_ = document_.append_child("fijee");
	// 
	fijee_.append_attribute("xmlns:fijee") = "https://github.com/Fijee-Project/Fijee";
     };
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
  XML_writer( const XML_writer& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class XML_writer
     */
    virtual ~XML_writer()
      {
	if( !memory_heavy_load_ )
	  document_.save_file( file_name_.c_str() ); 
      };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class XML_writer
     *
     */
    XML_writer& operator = ( const XML_writer& ){return *this;};

  public:
    /*!
     *
     *
     *
     */
    void set_file_name_( std::string File_Name ){file_name_ = File_Name;};
  };
}
#endif
