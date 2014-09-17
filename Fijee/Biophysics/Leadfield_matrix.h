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
#ifndef LEADFIELD_MATRIX_H
#define LEADFIELD_MATRIX_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Leadfield_matrix.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
#include <vector>
/*! \namespace Biophysics
 * 
 * Name space for our new package
 *
 */
namespace Biophysics
{
  /*! \class Leadfield_matrix
   * \brief class representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Leadfield_matrix
  {
  private:
    //! Electrode index
    int index_;
    //! Electrode label
    std::string label_;
    //! Potential from each dipole at the electrode
    std::vector< double > V_dipole_;
      

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Leadfield_matrix
     *
     */
    Leadfield_matrix();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Leadfield_matrix( const int, const std::string );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Leadfield_matrix( const Leadfield_matrix& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Leadfield_matrix( Leadfield_matrix&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Leadfield_matrix
     */
    virtual ~Leadfield_matrix(){/* Do nothing*/};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Leadfield_matrix
     *
     */
    Leadfield_matrix& operator = ( const Leadfield_matrix& );
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Leadfield_matrix
     *
     */
    Leadfield_matrix& operator = ( Leadfield_matrix&& );


  public:
    int get_index_() const {return index_;}
    std::string get_label_() const {return label_;}
    const std::vector< double >& get_V_dipole_() const {return V_dipole_;};
    // 
    void set_V_dipole(std::vector< double >&& Vd)
    { 
      V_dipole_.clear();
      V_dipole_ = std::move( Vd );
    }

  };
  /*!
   *  \brief Dump values for Leadfield_matrix
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Leadfield_matrix : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Leadfield_matrix& );
}
#endif
