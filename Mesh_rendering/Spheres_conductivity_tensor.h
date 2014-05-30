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
#ifndef SPHERES_CONDUCTIVITY_TENSOR_H_
#define SPHERES_CONDUCTIVITY_TENSOR_H_
/*!
 * \file Spheres_conductivity_tensor.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
//
// UCSF
//
#include "Utils/Fijee_environment.h"
#include "Conductivity_tensor.h"
#include "Cell_conductivity.h"
//
// Eigen
//
#include <Eigen/Dense>
// VTK
#include <vtkSmartPointer.h>
#include <vtkMatrix3x3.h>
//#include <vtkMath.h>

/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Spheres_conductivity_tensor
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Spheres_conductivity_tensor : public Conductivity_tensor
  {
  private:
    //! List of cell with matching conductivity coefficients
    std::list< Cell_conductivity > list_cell_conductivity_;


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Spheres_conductivity_tensor
     *
     */
    Spheres_conductivity_tensor();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Spheres_conductivity_tensor( const Spheres_conductivity_tensor& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Spheres_conductivity_tensor( Spheres_conductivity_tensor&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Spheres_conductivity_tensor
     */
    virtual ~Spheres_conductivity_tensor(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Spheres_conductivity_tensor
     *
     */
    Spheres_conductivity_tensor& operator = ( const Spheres_conductivity_tensor& );
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Spheres_conductivity_tensor
     *
     */
    Spheres_conductivity_tensor& operator = ( Spheres_conductivity_tensor&& );
    /*!
     *  \brief Move Operator ()
     *
     *  Object function for multi-threading
     *
     */
    virtual void operator ()()
    {
      Output_mesh_conductivity_xml();
    };

  public:
    /*!
     *  \brief Get number_of_pixels_x_
     *
     *  This method return the number of pixels x.
     *
     */
    ucsf_get_macro(list_cell_conductivity_, std::list< Cell_conductivity >);

  private:
    /*!
     */
    virtual void Make_analysis();

  public:
    /*!
     *  \brief Move_conductivity_array_to_parameters
     *
     *  This method moves members array to Access_Parameters's object.
     *
     */
    virtual void make_conductivity( const C3t3& );
    /*!
     *  \brief Output the XML match between mesh and conductivity
     *
     *  This method matches a conductivity tensor for each cell.
     *
     */
    virtual void Output_mesh_conductivity_xml();
    /*!
     *  \brief VTK visualization
     *
     *  This method gives a screenshot of the brain diffusion/conductivity vector field.
     *
     */
    void VTK_visualization();
  };
  /*!
   *  \brief Dump values for Spheres_conductivity_tensor
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Spheres_conductivity_tensor& );
};
#endif
