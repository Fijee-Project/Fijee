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
typedef CGAL::Surface_mesh_default_triangulation_3 Trs;
typedef Trs::Geom_traits GT;
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
