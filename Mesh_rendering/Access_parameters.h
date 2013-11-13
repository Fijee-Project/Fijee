#ifndef ACCESS_PARAMETERS_H_
#define ACCESS_PARAMETERS_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Access_parameters.hh
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <stdlib.h>     /* getenv */
#include <string>
#include <sstream>
#include <errno.h>    /* builtin errno*/
#include <sys/stat.h> /*mkdir*/
//
// NIFTI
//
#include "nifti1.h"
#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352
//
// Eigen
//
#include <Eigen/Dense>
typedef Eigen::Matrix< float, 3, 3 > Matrix_f_3X3;
typedef Eigen::Matrix< float, 3, 1 > Vector_f_3X1;
//
// Get built-in type.  Creates member get_"name"() (e.g., get_visibility());
//
#define ucsf_get_macro(name,type) \
  inline type get_##name() {	  \
    return this->name;		  \
  } 
//
// Get character string.  Creates member get_"name"() 
// (e.g., char *GetFilename());
//
#define ucsf_get_string_macro(name) \
  const char* get_##name() {	\
    return this->name.c_str();	\
  } 
//
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 * A singleton provides for encapsulation of persistent state information with the benefits of lazy construction. It also avoids the issue of which code segment should "own" the static persistent object instance. It further guarantees what mechanism is used to allocate memory for the underlying object and allows for better control over its destruction.
 *
 */
namespace Domains
{
  /*! \class Access_parameters
   *  \brief class representing whatever
   *
   *  This class is an example of singleton I will have to use
   */
  class Access_parameters
  {
  private:
    //! unique instance
    static Access_parameters* parameters_instance_;
    //! Freesurfer path. This path allow the program to reach all files we will need during the execution.
    std::string files_path_;
    std::string files_path_output_;

    //
    // Surface files
    //! Outer skin surface
    std::string outer_skin_surface_;
    //! Outer skull surface
    std::string outer_skull_surface_;
    //! Inner skull surface
    std::string inner_skull_surface_;
    //! Left hemisphere's pial
    std::string lh_pial_;
    //! Right hemisphere's pial
    std::string rh_pial_;
    //! Left hemisphere's white matter
    std::string lh_smoothwm_;
    //! Right hemisphere's white matter
    std::string rh_smoothwm_;
    //! All segmentation header 
    std::string aseg_hdr_;
    
    //
    // aseg.nii NIFTI information
    // image information
    //! number_of_pixels_x_ is number of pixels on X
    int number_of_pixels_x_;
    //! number_of_pixels_y_ is number of pixels on Y
    int number_of_pixels_y_;
    //! number_of_pixels_z_ is number of pixels on Z
    int number_of_pixels_z_;
    //! size_of_pixel_size_x_ is size of pixel on X
    float size_of_pixel_size_x_;
    //! size_of_pixel_size_y_ is size of pixel on Y
    float size_of_pixel_size_y_;
    //! size_of_pixel_size_z_ is size of pixel on Z
    float size_of_pixel_size_z_;
    //! Data array eigen values from nifti file
    float* data_eigen_values_;
    //! Data array eigen vector 1 from nifti file
    float* data_eigen_vector1_;
    //! Data array eigen vector 2 from nifti file
    float* data_eigen_vector2_;
    //! Data array eigen vector 3 from nifti file
    float* data_eigen_vector3_;
    // Transformation information
    //! rotation_ is the rotation matrix
    Matrix_f_3X3 rotation_;
    //! translation_ is the translation matrix
    Vector_f_3X1 translation_;
    
    //
    // NIFTI Diffusion/Conductivity data information
    //! number_of_pixels_x_ is number of pixels on X
    int eigenvalues_number_of_pixels_x_;
    //! number_of_pixels_y_ is number of pixels on Y
    int eigenvalues_number_of_pixels_y_;
    //! number_of_pixels_z_ is number of pixels on Z
    int eigenvalues_number_of_pixels_z_;
    //! eigenvalues_number_of_layers_ is number layers in the nifti format, e.g. 3 for three vector coordinates.
    int eigenvalues_number_of_layers_;
    //! size_of_pixel_size_x_ is size of pixel on X
    float eigenvalues_size_of_pixel_size_x_;
    //! size_of_pixel_size_y_ is size of pixel on Y
    float eigenvalues_size_of_pixel_size_y_;
    //! size_of_pixel_size_z_ is size of pixel on Z
    float eigenvalues_size_of_pixel_size_z_;
    // Transformation information
    //! rotation_ is the rotation matrix
    Matrix_f_3X3 eigenvalues_rotation_;
    //! translation_ is the translation matrix
    Vector_f_3X1 eigenvalues_translation_;

    //
    // Conductivity information
    //! Conductivity tensors array
    Eigen::Matrix <float, 3, 3>* conductivity_tensors_array_;
    //! eigen values matrices array
    Eigen::Matrix <float, 3, 3>* eigen_values_matrices_array_;
    //! Positions array of the conductivity tensor
    Eigen::Matrix <float, 3, 1>* positions_array_;
     //! Change base matrix array
    Eigen::Matrix <float, 3, 3>* P_matrices_array_;
   //! Speed up: check if we need make any calculation
    bool* Do_we_have_conductivity_; 


  protected:
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Access_parameters
     *
     */
    Access_parameters();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Access_parameters( const Access_parameters& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Access_parameters
     */
    virtual ~Access_parameters();
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Access_parameters
     *
     */
    Access_parameters& operator = ( const Access_parameters& );

  public:
    /*!
     *  \brief Get number_of_pixels_x_
     *
     *  This method return the number of pixels x.
     *
     */
    ucsf_get_macro(number_of_pixels_x_, int);
    /*!
     *  \brief Get number_of_pixels_y_
     *
     *  This method return the number of pixels y.
     *
     */
    ucsf_get_macro(number_of_pixels_y_, int);
    /*!
     *  \brief Get number_of_pixels_z_
     *
     *  This method return the number of pixels z.
     *
     */
    ucsf_get_macro(number_of_pixels_z_, int);
    /*!
     *  \brief Get size_of_pixel_size_x_
     *
     *  This method return the size of pixel size x.
     *
     */
    ucsf_get_macro(size_of_pixel_size_x_, int);
    /*!
     *  \brief Get size_of_pixel_size_y_
     *
     *  This method return the size of pixel size y.
     *
     */
    ucsf_get_macro(size_of_pixel_size_y_, int);
    /*!
     *  \brief Get size_of_pixel_size_z_
     *
     *  This method return the size of pixel size z.
     *
     */
    ucsf_get_macro(size_of_pixel_size_z_, int);
     /*!
     *  \brief Get rotation_
     *
     *  This method return the rotation matrix.
     *
     */
    ucsf_get_macro(rotation_, Matrix_f_3X3);
   /*!
     *  \brief Get translation_
     *
     *  This method return the translation vector.
     *
     */
    ucsf_get_macro(translation_, Vector_f_3X1);
   /*!
     *  \brief Get files_path_output_
     *
     *  This method return the output path
     *
     */
    ucsf_get_string_macro(files_path_output_);
    /*!
     *  \brief Get outer_skin_surface_
     *
     *  This method return the outer skin surface.
     *
     */
    ucsf_get_string_macro(outer_skin_surface_);
   /*!
     *  \brief Get outer_skull_surface_
     *
     *  This method return the outer skull surface.
     *
     */
    ucsf_get_string_macro(outer_skull_surface_);
   /*!
     *  \brief Get inner_skull_surface_
     *
     *  This method return the inner skull surface.
     *
     */
    ucsf_get_string_macro(inner_skull_surface_);
   /*!
     *  \brief Get lh_pial_
     *
     *  This method return the left hemisphere's pial.
     *
     */
    ucsf_get_string_macro(lh_pial_);
   /*!
     *  \brief Get rh_pial_
     *
     *  This method return the right hemisphere's pial.
     *
     */
    ucsf_get_string_macro(rh_pial_);
   /*!
     *  \brief Get lh_smoothwm_
     *
     *  This method return the left hemisphere's smooth white matter.
     *
     */
    ucsf_get_string_macro(lh_smoothwm_);
   /*!
     *  \brief Get rh_smoothwm_
     *
     *  This method return the right hemisphere's smooth white matter.
     *
     */
    ucsf_get_string_macro(rh_smoothwm_);
   /*!
     *  \brief Get aseg_hdr_
     *
     *  This method return the aseg header.
     *
     */
    ucsf_get_string_macro(aseg_hdr_);
    /*!
     *  \brief Get eigenvalues_number_of_pixels_x_
     *
     *  This method return the eigenvalues number of pixels x.
     *
     */
    ucsf_get_macro(eigenvalues_number_of_pixels_x_, int);
    /*!
     *  \brief Get eigenvalues_number_of_pixels_y_
     *
     *  This method return the eigenvalues number of pixels y.
     *
     */
    ucsf_get_macro(eigenvalues_number_of_pixels_y_, int);
    /*!
     *  \brief Get eigenvalues_number_of_pixels_z_
     *
     *  This method return the eigenvalues number of pixels z.
     *
     */
    ucsf_get_macro(eigenvalues_number_of_pixels_z_, int);
    /*!
     *  \brief Get eigenvalues_size_of_pixel_size_x_
     *
     *  This method return the eigenvalues size of pixel size x.
     *
     */
    ucsf_get_macro(eigenvalues_size_of_pixel_size_x_, int);
    /*!
     *  \brief Get eigenvalues_size_of_pixel_size_y_
     *
     *  This method return the eigenvalues size of pixel size y.
     *
     */
    ucsf_get_macro(eigenvalues_size_of_pixel_size_y_, int);
    /*!
     *  \brief Get eigenvalues_size_of_pixel_size_z_
     *
     *  This method return the eigenvalues size of pixel size z.
     *
     */
    ucsf_get_macro(eigenvalues_size_of_pixel_size_z_, int);
     /*!
     *  \brief Get eigenvalues_rotation_
     *
     *  This method return the eigenvalues rotation matrix.
     *
     */
    ucsf_get_macro(eigenvalues_rotation_, Matrix_f_3X3);
   /*!
     *  \brief Get eigenvalues_translation_
     *
     *  This method return the eigenvalues translation vector.
     *
     */
    ucsf_get_macro(eigenvalues_translation_, Vector_f_3X1);

  public:
    /*!
     *  \brief Get data_eigen_values_
     *
     *  This method move the nifti data eigen values into another array.
     *
     *  \param Data_Eigen_Values: destination array.
     */
    void get_data_eigen_values_( float** Data_Eigen_Values );
    /*!
     *  \brief Get data_eigen_vector1_
     *
     *  This method move the nifti data eigen vector1 into another array.
     *
     *  \param Data_Eigen_Vector1: destination array.
     */
    void get_data_eigen_vector1_( float** Data_Eigen_Vector1 );
    /*!
     *  \brief Get data_eigen_vector2_
     *
     *  This method move the nifti data eigen vector2 into another array.
     *
     *  \param Data_Eigen_Vector2: destination array.
     */
    void get_data_eigen_vector2_( float** Data_Eigen_Vector2 );
    /*!
     *  \brief Get data_eigen_vector3_
     *
     *  This method move the nifti data eigen vector3 into another array.
     *
     *  \param Data_Eigen_Vector3: destination array.
     */
    void get_data_eigen_vector3_( float** Data_Eigen_Vector3 );
    /*!
     *  \brief Get conductivity_tensors_array_
     *
     *  This method move the conductivity tensors array out of the parameters object.
     *
     *  \param Conductivity_Tensors_Array: conductivity tensors array from conductivity tensor object.
     */
    void get_conductivity_tensors_array_( Eigen::Matrix <float, 3, 3>** Conductivity_Tensors_Array );
    /*!
     *  \brief Get eigen_values_matrices_array_
     *
     *  This method move the eigen values matrices array out of the parameters object.
     *
     *  \param Eigen_Values_Matrices_Array: eigen values matrices array from conductivity object.
     */
    void get_eigen_values_matrices_array_( Eigen::Matrix <float, 3, 3>** Eigen_Values_Matrices_Array );
    /*!
     *  \brief Get positions_array_
     *
     *  This method move the positions array out of the parameters object.
     *
     *  \param Positions_Array: positions array from conductivity tensor object.
     */
    void get_positions_array_( Eigen::Matrix <float, 3, 1>** Positions_Array );
    /*!
     *  \brief Get P_matrices_array_
     *
     *  This method move the P matrices array out of the parameters object.
     *
     *  \param P_Matrices_Array: P matrices array from conductivity tensor object.
     */
    void get_P_matrices_array_( Eigen::Matrix <float, 3, 3>** P_Matrices_Array );
    /*!
     *  \brief Get Do_we_have_conductivity_
     *
     *  This method move the conductivity checking array out of the parameters object.
     *
     *  \param Do_We_Have_Conductivity: conductivity checking array from conductivity tensor object.
     */
    void get_Do_we_have_conductivity_( bool** Do_We_Have_Conductivity );
    /*!
     *  \brief Get the singleton instance
     *
     *  This method return the pointer parameters_instance_
     *
     */
    static Access_parameters* get_instance();
    /*!
     *  \brief Set conductivity_tensors_array_
     *
     *  This method move the conductivity tensors array inside the parameters object.
     *
     *  \param Conductivity_Tensors_Array: conductivity tensors array for mesh rendering object.
     */
    void set_conductivity_tensors_array_( Eigen::Matrix <float, 3, 3>** Conductivity_Tensors_Array );
    /*!
     *  \brief Set eigen_values_matrices_array_
     *
     *  This method move the conductivity tensors array inside the parameters object.
     *
     *  \param Eigen_Values_Matrices_Array: conductivity tensors array for mesh rendering object.
     */
    void set_eigen_values_matrices_array_( Eigen::Matrix <float, 3, 3>** Eigen_Values_Matrices_Array );
    /*!
     *  \brief Set positions_array_
     *
     *  This method move the positions array inside the parameters object.
     *
     *  \param Positions_Array: positions array for mesh rendering object.
     */
    void set_positions_array_( Eigen::Matrix <float, 3, 1>** Positions_Array );
    /*!
     *  \brief Set P_matrices_array_
     *
     *  This method move the P matrices array inside the parameters object.
     *
     *  \param P_Matrices_Array: P matrices array for mesh rendering object.
     */
    void set_P_matrices_array_( Eigen::Matrix <float, 3, 3>** P_Matrices_Array );
    /*!
     *  \brief Set Do_we_have_conductivity_
     *
     *  This method move the conductivity checking array inside the parameters object.
     *
     *  \param Do_We_Have_Conductivity: conductivity checking array for mesh rendering object.
     */
    void set_Do_we_have_conductivity_( bool** Do_We_Have_Conductivity );
    /*!
     *  \brief Kill the singleton instance
     *
     *  This method kill the singleton parameters_instance pointer
     *
     */
    static void kill_instance();
  };
  /*!
   *  \brief Dump values for Access_parameters
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Access_parameters& );
}
#endif
