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
#ifndef SPHERES_IMPLICITE_DOMAIN_H_
#define SPHERES_IMPLICITE_DOMAIN_H_
#include <iostream>
#include <memory>
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
#include "Access_parameters.h"
#include "Point_vector.h"
#include "Utils/enum.h"
#include "Implicite_domain.h"
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
//
//
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Spheres_implicite_domain.h
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
  /*! \class Spheres_implicite_domain
   * \brief classe building a list of "vertex, normal".
   *
   *  This class build a list of "vertex, normal" associated to a STL format mesh. This list will be an input of CGAL oracle for the construction of an implicite surface.
   * The implicite functions will help cearting the INRIMAGE format 3D image.  
   *
   */
  class Spheres_implicite_domain : public Implicite_domain
  {
  private:
    //! 2D mesh structure loaded in VTK functions
    std::string vtk_mesh_; 
    //! Implicite surface function
    std::unique_ptr< Poisson_reconstruction_function > function_;
    //! map of point with their vector. Those points represent vertices and their vectors (surface normal).
    std::list < Domains::Point_vector > point_normal_;
    //! VTK poly data bounds
    //! xmin = poly_data_bounds_[0]
    //! xmax = poly_data_bounds_[1]
    //! ymin = poly_data_bounds_[2]
    //! ymax = poly_data_bounds_[3]
    //! zmin = poly_data_bounds_[4]
    //! zmax = poly_data_bounds_[5]
    double poly_data_bounds_[6];

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Spheres_implicite_domain
     *
     */
    Spheres_implicite_domain();
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Spheres_implicite_domain
     *
     */
    Spheres_implicite_domain( const char* );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Spheres_implicite_domain( const Spheres_implicite_domain& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Spheres_implicite_domain( Spheres_implicite_domain&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Spheres_implicite_domain
     */
    virtual ~Spheres_implicite_domain(){ /* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Spheres_implicite_domain
     *
     */
    Spheres_implicite_domain& operator = ( const Spheres_implicite_domain& );
    /*!
     *  \brief Operator =
     *
     *  Move operator of the class Spheres_implicite_domain
     *
     */
    Spheres_implicite_domain& operator = ( Spheres_implicite_domain&& );
    /*!
     *  \brief Move Operator ()
     *
     *  Object function for multi-threading
     *
     */
    virtual void operator ()( double** );

  public:
    /*!
     *  \brief Get extrema values
     *
     *  This method return the extrema of the poly data point set.
     *
     *  \return extrema
     */
    virtual inline 
      const double* get_poly_data_bounds_() const {return poly_data_bounds_;};
    /*!
     *  \brief Get point_normal vector
     *
     *  This method return point_normal_ of the STL mesh.
     *
     *  \return point_normal_
     */
    virtual inline 
      std::list<Domains::Point_vector> get_point_normal_() const {return point_normal_;};
 
  public:
    virtual inline 
      bool inside_domain( CGAL::Point_3< Kernel > Point )
    {
      return ( (*function_)( Point ) <= 0. ? true : false );
    };
    
  };
  /*!
   *  \brief Dump values for Spheres_implicite_domain
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Spheres_implicite_domain& );
};
#endif
