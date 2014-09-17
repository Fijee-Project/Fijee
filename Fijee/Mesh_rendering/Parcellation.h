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
#ifndef PARCELLATION_H
#define PARCELLATION_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Parcellation.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
// UCSF
//
#include "Utils/enum.h"
#include "Utils/Statistical_analysis.h"
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Parcellation
   * \brief classe representing whatever
   *
   *  This class is the mother class for head parcellation process.
   */
  class Parcellation : public Utils::Statistical_analysis
  {
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Parcellation
     *
     */
    Parcellation();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Parcellation( const Parcellation& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Parcellation
     */
    virtual ~Parcellation();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Parcellation
     *
     */
    Parcellation& operator = ( const Parcellation& );
     /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Parcellation
     *
     */
    virtual void operator ()() = 0;
   
  public:
    /*!
     */
    virtual void Mesh_partitioning() = 0;
    /*!
     */
    virtual long int get_region( int ) = 0;
    /*!
    */
    virtual bool check_partitioning( int  ) = 0;

  private:
    /*!
     */
    virtual void Make_analysis() = 0;
};
  /*!
   *  \brief Dump values for Parcellation
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Parcellation& );
};
#endif
