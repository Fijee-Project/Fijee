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
#ifndef CONDUCTIVITY_TENSOR_H_
#define CONDUCTIVITY_TENSOR_H_
//
// UCSF
//
#include "Utils/Statistical_analysis.h"
#include "Utils/Fijee_environment.h"
#include "Utils/enum.h"
#include "CGAL_tools.h"
/*!
 * \file Conductivity_tensor.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Conductivity_tensor
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Conductivity_tensor : public Utils::Statistical_analysis
  {
  public:
    virtual ~Conductivity_tensor(){/* Do nothing */};

    /*!
     *  \brief Operator ()
     *
     *  Object function for multi-threading
     *
     */
    virtual void operator ()() = 0;
//
//  private:
//    /*!
//     *  \brief Move_conductivity_array_to_parameters
//     *
//     *  This method moves members array to Access_Parameters's object.
//     *
//     */
//    void move_conductivity_array_to_parameters();
//
  public:
    /*!
     */
    virtual void Make_analysis() = 0;
    /*!
     *  \brief Move_conductivity_array_to_parameters
     *
     *  This method moves members array to Access_Parameters's object.
     *
     */
    virtual void make_conductivity( const C3t3& ) = 0;
    /*!
     *  \brief Output the XML match between mesh and conductivity
     *
     *  This method matches a conductivity tensor for each cell.
     *
     */
    virtual void Output_mesh_conductivity_xml() = 0;
//    /*!
//     *  \brief VTK visualization
//     *
//     *  This method gives a screenshot of the brain diffusion/conductivity vector field.
//     *
//     */
//    void VTK_visualization();
//    /*!
//     *  \brief 
//     *
//     *  This method 
//     *
//     */
//    void INRIMAGE_image_of_conductivity_anisotropy();
  };
};
#endif
