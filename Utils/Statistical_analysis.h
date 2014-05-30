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
#ifndef STATISTICAL_ANALYSIS_H_
#define STATISTICAL_ANALYSIS_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Statistical_analysis.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <fstream>      // std::ifstream, std::ofstream
#include <sstream>
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Statistical_analysis
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Statistical_analysis
  {
  private:
    std::ofstream file_;


  protected:
    //! Stream populating the output file.
    std::stringstream output_stream_;

    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Statistical_analysis
     *
     */
    Statistical_analysis(){};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Statistical_analysis( const Statistical_analysis& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Statistical_analysis
     */
    virtual ~Statistical_analysis(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Statistical_analysis
     *
     */
    Statistical_analysis& operator = ( const Statistical_analysis& ){return *this;};
    
  public:
    /*!
     */
    void Make_output_file( const char* file_name)
    {
      file_.open( file_name );
      file_ << output_stream_.rdbuf();
      file_.close();  
    };

  private:
    /*!
     */
    virtual void Make_analysis() = 0;
 };
  /*!
   *  \brief Dump values for Statistical_analysis
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Statistical_analysis& );
};
#endif
