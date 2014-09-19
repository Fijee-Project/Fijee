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
#ifndef BUILD_DIPOLES_LIST_H
#define BUILD_DIPOLES_LIST_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Build_dipoles_list.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <list>
//
// UCSF
//
#include "Fijee/Fijee_enum.h"
#include "Point_vector.h"
#include "Dipole.h"
#include "Cell_conductivity.h"
#include "Fijee/Fijee_statistical_analysis.h"
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Build_dipoles_list
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Build_dipoles_list : public Fijee::Statistical_analysis
  {
  private:

    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Build_dipoles_list
     *
     */
    Build_dipoles_list();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Build_dipoles_list( const Build_dipoles_list& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Build_dipoles_list
     */
    virtual ~Build_dipoles_list();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Build_dipoles_list
     *
     */
    Build_dipoles_list& operator = ( const Build_dipoles_list& );
     /*!
     *  \brief Operator ()
     *
     *  Operator () of the class Build_dipoles_list
     *
     */
    virtual void operator () () = 0;
   
  public:
    /*!
     */
    virtual void Make_list( const std::list< Cell_conductivity >& List_cell_conductivity ) = 0;
    /*!
     *  \brief Output the XML of the dipoles' list
     *
     *  This method create the list of dipoles.
     *
     */
    virtual void Output_dipoles_list_xml() = 0;
    /*!
     *  \brief Output the XML of the parcellation' list
     *
     *  This method create the list of parcellation dipoles.
     *
     */
    virtual void Output_parcellation_list_xml() = 0;

  private:
    /*!
     */
    virtual void Make_analysis() = 0;
    /*!
     *  \brief Build stream
     *
     *  This method create the output stream.
     *
     */
    virtual void Build_stream(const std::list< Domains::Dipole >&, std::ofstream&) = 0;
    /*!
     */
    virtual void Parcellation_list() = 0;
};
  /*!
   *  \brief Dump values for Build_dipoles_list
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Build_dipoles_list& );
};
#endif
