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
//#include <stdlib.h>     /* getenv */
#include <string>
#include <sstream>
//#include <errno.h>      /* builtin errno */
//#include <sys/stat.h>   /* mkdir */
#include <list>
#include <tuple>
#include <algorithm>    /* copy */
//
// UCSF
//
#include "Utils/Fijee_environment.h"
#include "Point_vector.h"
#include "Distance.h"
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_with_normal_3.h>

//#include <CGAL/Simple_cartesian.h>
//#include <CGAL/Orthogonal_incremental_neighbor_search.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
//#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits.h>
//
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;

typedef CGAL::Search_traits<float, Domains::Point_vector, const float*, Construct_coord_iterator> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits, Domains::Distance > Neighbor_search;
typedef Neighbor_search::iterator NN_iterator;
typedef Neighbor_search::Tree Tree;

//typedef CGAL::Search_traits_3<Kernel> TreeTraits;
//typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
////typedef CGAL::Orthogonal_incremental_neighbor_search<TreeTraits> NN_incremental_search;
//typedef Neighbor_search::iterator NN_iterator;
//typedef Neighbor_search::Tree Tree;
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
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 * 
 *
 */
namespace Domains
{
  /*! \class Access_parameters
   *  \brief class representing whatever
   *
   *  This class provides for encapsulation of persistent state information. It also avoids the issue of which code segment should "own" the static persistent object instance. It further guarantees what mechanism is used to allocate memory for the underlying object and allows for better control over its destruction.
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
    // Surfaces contenairs
    //! List of point with their vector for the gray matter left hemisphere
    std::list< Domains::Point_vector > lh_gray_matter_surface_point_normal_;
    //! List of point with their vector for the gray matter write hemisphere
    std::list< Domains::Point_vector > rh_gray_matter_surface_point_normal_;
    //! List of point with their vector for the white matter left hemisphere
    std::list< Domains::Point_vector > lh_white_matter_surface_point_normal_;
    //! List of point with their vector for the white matter write hemisphere
    std::list< Domains::Point_vector > rh_white_matter_surface_point_normal_;
    //! List of matching vertices between white matter and gray matter left hemisphere
    std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > > lh_match_wm_gm_;
    //! List of matching vertices between white matter and gray matter right hemisphere
    std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > > rh_match_wm_gm_;

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
    // Delta between our transformation and MNI 305 transformation
    //! Delta rotation_ is the rotation matrix
    Matrix_f_3X3 delta_rotation_;
    //! Delta translation_ is the translation matrix
    Vector_f_3X1 delta_translation_;
    
    
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
    // Electrodes
    //! Electrodes Standard 10/20 Cap81
    std::string electrodes_10_20_;


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
     *  \brief Get delta_rotation_
     *
     *  This method return the delta rotation matrix between nifti rotation and MNI305 rotation.
     *
     */
    ucsf_get_macro(delta_rotation_, Matrix_f_3X3);
   /*!
     *  \brief Get delta_translation_
     *
     *  This method return the delta translation vector between nifti rotation and MNI305 rotation.
     *
     */
    ucsf_get_macro(delta_translation_, Vector_f_3X1);
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
   /*!
     *  \brief Get lh_match_wm_gm_
     *
     *  This method return the white and gray matter vertices matching tuples for the left hemisphere.
     *
     */
    void get_lh_match_wm_gm_(std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > >& Lh_match_wm_gm) const 
    { 
      Lh_match_wm_gm.resize( lh_match_wm_gm_.size() );
      //
      std::move( lh_match_wm_gm_.begin(), lh_match_wm_gm_.end(), Lh_match_wm_gm.begin() );
    };
   /*!
     *  \brief Get rh_match_wm_gm_
     *
     *  This method return the white and gray matter vertices matching tuples for the right hemisphere.
     *
     */
    void get_rh_match_wm_gm_(std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > >& Rh_match_wm_gm) const 
    {  
      Rh_match_wm_gm.resize( rh_match_wm_gm_.size() );
      //
      std::move( rh_match_wm_gm_.begin(), rh_match_wm_gm_.end(), Rh_match_wm_gm.begin() );
    };
   /*!
     *  \brief Get electrodes standard 10/20
     *
     *  This method return the electrodes standard 10/20 list.
     *
     */
    ucsf_get_string_macro(electrodes_10_20_);


  public:
     /*!
     *  \brief Set rotation_ along with MNI305
     *
     *  This method reset the rotation to the MNI305 rotation matrix.
     *
     */
    inline void set_rotation_MNI305_()
    { 
      rotation_ <<
	-1., 0., 0.,
	 0., 0., 1.,
	 0.,-1., 0.;
    };
   /*!
     *  \brief Set translation_
     *
     *  This method set the translation vector to the MNI305 translation vector.
     *
     */
    inline void set_translation_MNI305_()
    {
      translation_  <<
	 128.,
	-128.,
	 128.;
    };
     /*!
     *  \brief Set rotation_ Id
     *
     *  This method reset the rotation to Id matrix.
     *
     */
    inline void set_rotation_Id_()
    { 
      rotation_ <<
	1., 0., 0.,
	0., 1., 0.,
	0., 0., 1.;
    };
   /*!
     *  \brief Set translation_ as 128 vector
     *
     *  This method set the translation vector as 128 mm translation vector.
     *
     */
    inline void set_translation_128_()
    {
      translation_  <<
	128.,
	128.,
	128.;
    };
     /*!
     *  \brief Set delta_rotation_
     *
     *  This method set the delta rotation matrix between nifti rotation and MNI305 rotation.
     *
     */
    inline void set_delta_rotation_(Matrix_f_3X3 Delta_Rotation){delta_rotation_ = Delta_Rotation;};
   /*!
     *  \brief Set delta_translation_
     *
     *  This method set the delta translation vector between nifti translation and MNI305 translation.
     *
     */
    inline void set_delta_translation_(Vector_f_3X1 Delta_Translation){delta_translation_ = Delta_Translation;};
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
     *  \brief Get the singleton instance
     *
     *  This method return the pointer parameters_instance_
     *
     */
    static Access_parameters* get_instance();
    /*!
     *  \brief Set lh_gray_matter_surface_point_normal_
     *
     *  This method move the left hemisphere gray matter surface point normal vector inside the parameters object.
     *
     *  \param Lh_gray_matter_surface_point_normal: gray matter surface point normal vector of the gray matter surface mesh.
     */
    void set_lh_gray_matter_surface_point_normal_(std::list<Point_vector>&&  Lh_gray_matter_surface_point_normal );
    /*!
     *  \brief Set rh_gray_matter_surface_point_normal_
     *
     *  This method move the right hemisphere gray matter surface point normal vector inside the parameters object.
     *
     *  \param Rh_gray_matter_surface_point_normal: gray matter surface point normal vector of the gray matter surface mesh.
     */
    void set_rh_gray_matter_surface_point_normal_(std::list<Point_vector>&&  Rh_gray_matter_surface_point_normal );
    /*!
     *  \brief Set lh_white_matter_surface_point_normal_
     *
     *  This method move the left hemisphere white matter surface point normal vector inside the parameters object.
     *
     *  \param White_matter_surface_point_normal: white matter surface point normal vector of the white matter surface mesh.
     */
    void set_lh_white_matter_surface_point_normal_(std::list<Point_vector>&&  Lh_white_matter_surface_point_normal );
    /*!
     *  \brief Set rh_white_matter_surface_point_normal_
     *
     *  This method move the right hemisphere white matter surface point normal vector inside the parameters object.
     *
     *  \param White_matter_surface_point_normal: white matter surface point normal vector of the white matter surface mesh.
     */
    void set_rh_white_matter_surface_point_normal_(std::list<Point_vector>&&  Rh_white_matter_surface_point_normal );
    /*!
     *  \brief Kill the singleton instance
     *
     *  This method kill the singleton parameters_instance pointer
     *
     */
    static void kill_instance();
    /*!
     *  \brief init parameters
     *
     *  This method initialize the simulation parameters.
     *
     */
    void init(){};
    /*!
     *  \brief epitaxy growth
     *
     *  This method method matches the white matter vertices with the gray matter vertices. 
     *  The goal is not only finding the closest vertices from the two materials but also minimize the angle between the carried normals. Most likely the gray matter follows an epitaxy growth.
     *
     */
    void epitaxy_growth();

  private:
    /*!
     *  \brief matching vertices between gray and white matters
     *
     *  This method method creates a list of tuples match of vertices between gray and white matter.
     * 
     *  \param White_matter_surface_point_normal: vector of white matter vertices with their normal.
     *  \param Gray_matter_surface_point_normal: vector of gray matter vertices with their normal.
     *  \param Match_wm_gm: list of tuple matching white matter with gray matter vertices.
     */
    void match_wm_gm( std::list<Point_vector>&  White_matter_surface_point_normal, 
		      std::list<Point_vector>&  Gray_matter_surface_point_normal, 
		      std::list< std::tuple< Point_vector, Point_vector > >& Match_wm_gm );
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
