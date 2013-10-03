#ifndef VTK_IMPLICITE_DOMAIN_H_
#define VTK_IMPLICITE_DOMAIN_H_
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>      // std::stringstream
#include <map>
#include <set>
#include <vector>
//
// UCSF
//
#include "Point_vector.h"
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/compute_average_spacing.h>
//
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;
typedef CGAL::Poisson_reconstruction_function<Kernel> Poisson_reconstruction_function;
typedef Kernel::FT FT;
//
// VTK
//
#include <vtkSmartPointer.h>
//
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include "vtkPolyDataAlgorithm.h"
#include <vtkSelectEnclosedPoints.h>
// Rendering
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPointData.h>
// Transformation
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
// OBBTree Data structure
#include <vtkPoints.h>
#include <vtkLine.h>
#include <vtkOBBTree.h>
//
//
//
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file VTK_implicite_domain.h
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
  /*! \class VTK_implicite_domain
   * \brief classe building a list of "vertex, normal".
   *
   *  This class build a list of "vertex, normal" associated to a STL format mesh. This list will be an input of CGAL oracle for the construction of an implicite surface.
   * The implicite functions will help cearting the INRIMAGE format 3D image.  
   *
   */
  class VTK_implicite_domain
  {
  private:
    std::string vtk_mesh_; 
    //! Implicite surface function
    Poisson_reconstruction_function* function_;
    //
    vtkSmartPointer<vtkOBBTree> obb_tree_;
    vtkSmartPointer<vtkPoints>  check_points_;
    vtkSmartPointer<vtkSelectEnclosedPoints> select_enclosed_points_;
    //! map of point with their vector. Those points represent vertices and their vectors represent the surface normal.
    std::list <Domains::Point_vector > point_normal_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class VTK_implicite_domain
     *
     */
    VTK_implicite_domain();
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class VTK_implicite_domain
     *
     */
    VTK_implicite_domain( const char* );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    VTK_implicite_domain( const VTK_implicite_domain& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    VTK_implicite_domain( VTK_implicite_domain&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class VTK_implicite_domain
     */
    virtual ~VTK_implicite_domain();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class VTK_implicite_domain
     *
     */
    VTK_implicite_domain& operator = ( const VTK_implicite_domain& );
    /*!
     *  \brief Operator =
     *
     *  Move operator of the class VTK_implicite_domain
     *
     */
    VTK_implicite_domain& operator = ( VTK_implicite_domain&& );
    /*!
     *  \brief Move Operator ()
     *
     *  Object function for multi-threading
     *
     */
    void operator ()( double** );

  public:
    /*!
     *  \brief Get max_x_ value
     *
     *  This method return the maximum x coordinate max_x_.
     *
     *  \return max_x_
     */
    //    inline double get_max_x( ) const {return max_x_;};
    /*!
     *  \brief Get point_normal vector
     *
     *  This method return point_normal_ of the STL mesh.
     *
     *  \return point_normal_
     */
    inline  std::list<Domains::Point_vector> get_point_normal_( ) const {return point_normal_;};
 
  public:
    inline bool inside_domain( CGAL::Point_3< Kernel > Point )
    {
      return ( (*function_)( Point ) <= 0. ? true : false );
    };

  };
  /*!
   *  \brief Dump values for VTK_implicite_domain
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const VTK_implicite_domain& );
};
#endif
