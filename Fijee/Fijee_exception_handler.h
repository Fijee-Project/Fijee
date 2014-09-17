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
#ifndef FIJEE_EXCEPTION_HANDLER_H
#define FIJEE_EXCEPTION_HANDLER_H
/*!
 * \file Fijee_exception_handler.h
 * \brief Exception management 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <errno.h>      /* builtin errno */
#include <string>
#include <exception>
#include <cstdlib>
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
  /*! \class Exception_handler
   *
   * \brief class representing exception handling.
   *
   *  This class is the generic exception class. By deriving all of our exceptions from this base class, we inssure a generic the output message handling.
   */
  class Exception_handler
  {
  protected:
    //! Exception message
    std::string exception_message_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Exception_handler
     *
     */
  Exception_handler():
    exception_message_(std::string("")){};
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Exception_handler
     *
     */
  Exception_handler( const std::string& Exception_message ):
    exception_message_(Exception_message){};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
  Exception_handler( const Exception_handler& that):
    exception_message_(that.exception_message_){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Exception_handler
     */
    virtual ~Exception_handler(){/*Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Exception_handler
     *
     */
    Exception_handler& operator = ( const Exception_handler& that)
      {
	exception_message_ = that.what();
	
	// 
	return *this;
      };

  public:
    /*!
     *  \brief Get exception_message_
     *
     *  This methode access exception_message_ member.
     *
     */
    std::string what()const{return exception_message_;};
  };

  /*! \class Exit_handler
   * \brief classe representing exception handling.
   *
   *  This class is the exit hadling class. It allows a simple exit class without coredump.
   *
   */
  class Exit_handler : public Exception_handler
  {
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Exit_handler
     *
     */
  Exit_handler(): Exception_handler(){};
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Exit_handler
     *
     */
    Exit_handler( const std::string& Exception_message,
		  const int Line, const std::string& File )
    {
      exception_message_ = std::string("\n\n=== Fijee exit handler called ===\n");
      exception_message_ += std::string("Message: \n");
      exception_message_ += Exception_message;
      exception_message_ += std::string("\nAt line: ")+ std::to_string(Line);
      exception_message_ += std::string(" in source: ") + File;
      exception_message_ += std::string("\n");
    };
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
  Exit_handler( const Exit_handler& that):
    Exception_handler(that){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Exit_handler
     */
    virtual ~Exit_handler()
      {
	exit( EXIT_FAILURE );
      };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Exit_handler
     *
     */
    Exit_handler& operator = ( const Exit_handler& that)
      {
	Exception_handler::operator = (that);
	// 
	return *this;
      };
  };

  /*! \class Error_handler
   * \brief classe representing exception handling.
   *
   *  This class is the error handling class. It generates core dump.
   */
  class Error_handler : public Exception_handler
  {
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Error_handler
     *
     */
  Error_handler(): Exception_handler(){};
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Error_handler
     *
     */
    Error_handler( const std::string& Exception_message,
		  const int Line, const std::string& File )
    {
      exception_message_ = std::string("\n\n=== Fijee error handler called ===\n");
      exception_message_ += std::string("Message: \n");
      exception_message_ += Exception_message;
      exception_message_ += std::string("\nAt line: ") + std::to_string(Line);
      exception_message_ += std::string(" in source: ") + File;
      exception_message_ += std::string("\n");
    };
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
  Error_handler( const Error_handler& that):
    Exception_handler(that){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Error_handler
     */
    virtual ~Error_handler()
      {
	abort();
      };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Error_handler
     *
     */
    Error_handler& operator = ( const Error_handler& that)
      {
	Exception_handler::operator = (that);
	// 
	return *this;
      };
  };
}
#endif
