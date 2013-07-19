#ifndef CGAL_IMPLICITE_DOMAIN_H_
#define CGAL_IMPLICITE_DOMAIN_H_
#include <iostream>
#include <string>
#include <fstream>
//
// CGAL
//
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Surface_mesh_default_criteria_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/IO/Complex_2_in_triangulation_3_file_writer.h>
// implicite surface construction
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Gray_level_image_3.h>
#include <CGAL/Implicit_surface_3.h>
//
// default triangulation for Surface_mesher
typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef Tr::Geom_traits GT;
typedef CGAL::Gray_level_image_3<GT::FT, GT::Point_3> Gray_level_image;

//
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file CGAL_implicite_domain.h
 * \brief brief explaination 
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
  /*! \class CGAL_implicite_domain
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class CGAL_implicite_domain
  {
  private:
    std::string binary_mask_; 
    //
    CGAL::Implicit_surface_3<GT, Gray_level_image>* select_enclosed_points_;
//    Poisson_reconstruction_function* function_;
//    FT average_spacing_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class CGAL_implicite_domain
     *
     */
    CGAL_implicite_domain();
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class CGAL_implicite_domain
     *
     */
    CGAL_implicite_domain( const char* );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    CGAL_implicite_domain( const CGAL_implicite_domain& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    CGAL_implicite_domain( CGAL_implicite_domain&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class CGAL_implicite_domain
     */
    virtual ~CGAL_implicite_domain();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class CGAL_implicite_domain
     *
     */
    CGAL_implicite_domain& operator = ( const CGAL_implicite_domain& );
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class CGAL_implicite_domain
     *
     */
    CGAL_implicite_domain& operator = ( CGAL_implicite_domain&& );
    /*!
     *  \brief Move Operator ()
     *
     *  Move operator of the class CGAL_implicite_domain
     *
     */
    void operator ()( float** );

  public:
    /*!
     *  \brief Get max_x_ value
     *
     *  This method return the maximum x coordinate max_x_.
     *
     *  \return max_x_
     */
    //    inline double get_max_x( ) const {return max_x_;};
 
  public:
    /*!
     *  \brief Get max_x_ value
     *
     *  This method return the maximum x coordinate max_x_.
     *
     *  \return max_x_
     */
    inline bool inside_domain( GT::Point_3 Position )
    {
      return ( (*select_enclosed_points_)( Position ) == -1 ? true : false);
    };

  };
  /*!
   *  \brief Dump values for CGAL_implicite_domain
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const CGAL_implicite_domain& );
};
#endif
