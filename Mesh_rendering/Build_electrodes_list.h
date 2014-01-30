#ifndef BUILD_ELECTRODES_LIST_H_
#define BUILD_ELECTRODES_LIST_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Build_electrodes_list.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <list>
//
// UCSF
//
#include "Utils/enum.h"
#include "Access_parameters.h"
#include "Electrode.h"
#include "Labeled_domain.h"
#include "VTK_implicite_domain.h"
#include "Spheres_implicite_domain.h"
//#include "Point_vector.h"
#include "Utils/Statistical_analysis.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
////
//// VTK
////
//#include <vtkSmartPointer.h>
//#include <vtkTimerLog.h>
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Mesh_3/Image_to_labeled_function_wrapper.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
// Implicite functions
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;
typedef CGAL::Surface_mesh_default_triangulation_3 Triangle_surface;
typedef Triangle_surface::Geom_traits GT;
typedef CGAL::Mesh_3::Image_to_labeled_function_wrapper<CGAL::Image_3, Kernel > Image_wrapper;//
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Build_electrodes_list
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Build_electrodes_list
  {
  private:
    //! List of electrods
    std::list< Domains::Electrode > electrodes_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Build_electrodes_list
     *
     */
    Build_electrodes_list();
//    /*!
//     *  \brief Copy Constructor
//     *
//     *  Constructor is a copy constructor
//     *
//     */
//    Build_electrodes_list( const Build_electrodes_list& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Build_electrodes_list
     */
    ~Build_electrodes_list(){/* Do nothing */};
//    /*!
//     *  \brief Operator =
//     *
//     *  Operator = of the class Build_electrodes_list
//     *
//     */
//    Build_electrodes_list& operator = ( const Build_electrodes_list& );

  public:
    /*!
     *  \brief adjust the electrod position on a surface
     *
     *  Adjust the electrod position on the surface Surf in the spheres model case.
     *
     */
    void adjust_cap_positions_on( Labeled_domain< Spheres_implicite_domain, 
				                  GT::Point_3, 
				                  std::list< Point_vector > >&  Surf );    
    /*!
     *  \brief adjust the electrod position on a surface
     *
     *  Adjust the electrod position on the surface Surf1 using the center of coordinate from the surface Surf2.
     *
     */
    void adjust_cap_positions_on( Labeled_domain< VTK_implicite_domain, 
				                  GT::Point_3, 
				                  std::list< Point_vector > >&  Surf1,
				  Labeled_domain< VTK_implicite_domain, 
				                  GT::Point_3, 
				                  std::list< Point_vector > >&  Surf2 );
    /*!
     *  \brief inside the domaine
     *
     *  This function check if the point Point is inside an electrode.
     *
     * \param Point: check point.
     *
     */
    bool inside_domain( GT::Point_3 Point );
    /*!
     *  \brief Output the XML of the electrodes' list
     *
     *  This method create the list electrodes
     *
     */
    void Output_electrodes_list_xml();
    /*!
     *  \brief Build stream
     *
     *  This method create the output stream.
     *
     */
    void Build_stream(std::ofstream&);
  };
  /*!
   *  \brief Dump values for Build_electrodes_list
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Build_electrodes_list& );
};
#endif
