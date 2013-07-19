#ifndef CONDUCTIVITY_TENSOR_H_
#define CONDUCTIVITY_TENSOR_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Conductivity_tensor.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
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
  /*! \class Conductivity_tensor
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Conductivity_tensor
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
    //! Mapping between nifti diffusion data index and aseg nifti data index
    int* nifti_data_to_diffusion_mapping_array_;

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

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Conductivity_tensor
     *
     */
    Conductivity_tensor();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Conductivity_tensor( const Conductivity_tensor& ) = delete;
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Conductivity_tensor( Conductivity_tensor&& ) = delete;
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Conductivity_tensor
     */
    virtual ~Conductivity_tensor();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Conductivity_tensor
     *
     */
    Conductivity_tensor& operator = ( const Conductivity_tensor& ) = delete;
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Conductivity_tensor
     *
     */
    Conductivity_tensor& operator = ( Conductivity_tensor&& ) = delete;
    /*!
     *  \brief Move Operator ()
     *
     *  Object function for multi-threading
     *
     */
    void operator ()();

  public:
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
   *  \brief Dump values for Conductivity_tensor
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Conductivity_tensor& );
};
#endif
