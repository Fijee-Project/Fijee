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
#ifndef HEAD_CONDUCTIVITY_TENSOR_H
#define HEAD_CONDUCTIVITY_TENSOR_H
/*!
 * \file Head_conductivity_tensor.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
#include <memory>
#include <thread>
//
// UCSF
//
#include "Utils/Fijee_environment.h"
#include "Conductivity_tensor.h"
#include "CGAL_tools.h"
#include "Cell_conductivity.h"

#include <metis.h>
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
  /*! \class Head_conductivity_tensor
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Head_conductivity_tensor : public Conductivity_tensor
  {
  private:
    //
    // Transformation from diffusion framework to aseg/orig framework
    // Frameworks aseg/orig are, in fine, the mesh rendering framework
    //
    //! Rotation matrix from diffusion framework to aseg/orig framework
    Eigen::Matrix< float, 3, 3> rotation_mesh_framework_;
    //! Translation vector from diffusion framework to aseg/orig framework
    Eigen::Matrix< float, 3, 1> translation_mesh_framework_;

    //
    // Nifti image information
    //
    //! Number of pixels X
    int number_of_pixels_x_;
    //! Number of pixels Y
    int number_of_pixels_y_;
    //! Number of pixels Z
    int number_of_pixels_z_;
    //! Size of pixel X
    float size_of_pixel_size_x_;
    //! Size of pixel Y
    float size_of_pixel_size_y_;
    //! Size of pixel Z
    float size_of_pixel_size_z_;
    //! Rotation matrix
    Eigen::Matrix< float, 3, 3 > rotation_;
    //! Translation matrix
    Eigen::Matrix< float, 3, 1 > translation_;
    //! scaling factor
    int qfac_;
 
    //
    // Nifti image retrieved and traited information
    //
    //! Eigen values diagonal matrices array
    Eigen::Matrix <float, 3, 3>* eigen_values_matrices_array_;
    //! Change of basis matrices array
    Eigen::Matrix <float, 3, 3>* P_matrices_array_;
    //! Conductivity tensors array
    Eigen::Matrix <float, 3, 3>* conductivity_tensors_array_;
    //! Positions array of the conductivity tensor
    Eigen::Matrix <float, 3, 1>* positions_array_;
    //! Speed up: check if we need make any calculation
    bool* Do_we_have_conductivity_; 

    //
    // Conductivity matching
    //
    //! List of cell with matching conductivity coefficients
    std::list< Cell_conductivity > list_cell_conductivity_;


  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Head_conductivity_tensor
     *
     */
    Head_conductivity_tensor();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Head_conductivity_tensor( const Head_conductivity_tensor& ) = delete;
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Head_conductivity_tensor( Head_conductivity_tensor&& ) = delete;
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Head_conductivity_tensor
     */
    virtual ~Head_conductivity_tensor();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Head_conductivity_tensor
     *
     */
    Head_conductivity_tensor& operator = ( const Head_conductivity_tensor& ) = delete;
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Head_conductivity_tensor
     *
     */
    Head_conductivity_tensor& operator = ( Head_conductivity_tensor&& ) = delete;
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
     *  \brief 
     *
     *  This method 
     *
     */
    void Make_analysis();

  public:
    /*!
     *  \brief Make conductivity
     *
     *  This method compute the conductivity at mesh centroids.
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
    /*!
     *  \brief 
     *
     *  This method 
     *
     */
    void INRIMAGE_image_of_conductivity_anisotropy();
  };
  /*!
   *  \brief Dump values for Head_conductivity_tensor
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Head_conductivity_tensor& );
};
#endif
