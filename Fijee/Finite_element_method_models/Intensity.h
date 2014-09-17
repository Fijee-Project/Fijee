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
#ifndef _INTENSITY_H
#define _INTENSITY_H
#include <dolfin.h>
#include <vector>
#include <string>
#include <complex>
//
//
//
using namespace dolfin;
/*!
 * \file Intensity.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Solver
{
  /*! \class Intensity
   * \brief classe representing an electrode intensity
   *
   *  This class implements the geometrical caracteristics of an electrode: shape, size, ... Intensity also implements the current injection and the measured/simulated potential.
   */
  class Intensity// : public Expression
  {
  private:
    //
    // Physical properties
    //
    //! Electric variable
    std::string electric_variable_;
    //! Electrode index
    int index_;
    //! Electrode position
    Point r0_values_;  
    //! Electrode position projected on the closest boundary vertex
    Point r0_projection_;  
    //! Index of the electrode position projected vertex
    std::size_t r0_projection_index_;
    //! Electrode direction vector
    Point e_values_;  
    //! Electrode intensity [I_] = A
    double I_;
    //! Electrode name
    std::string label_;
    //! Electrode impedance
    std::complex<double> impedance_;
    //! Contact suface between the electrode and the scalp [S] = mm^2
    double surface_;
    //! Contact suface radius between the electrode and the scalp
    double radius_;

    //
    // Geometry propterties
    //
    //! map of cells on boundaries with their facets effectivily on the boundaries
    std::map< std::size_t, std::list< MeshEntity  >  > boundary_cells_;
    //! Set of boundaries mesh cells
    std::set<std::size_t> boundary_vertices_;
    //! Imposed one source point. It is used for the validation comparison
    mutable bool not_yet_;

    mutable std::set<Point> points_Dirac_;
    mutable std::list<std::size_t> facet_reservoir_;

    //
    // Electrode electrical potential calculation
    // 
    //! Type of potential return true if the potential evaluation is punctual, false otherwise.
    bool type_of_potential_;
    //! Electrical potential list
    std::list< double > electrical_potential_list_;
    //! Electrical potential
    double V_;
   

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Intensity
     *
     */
    Intensity();
    /*!
     *  \brief Copy Constructor
     *
     *  Copy constructor of the class Intensity
     *
     */
    Intensity( const Intensity& );
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Intensity
     *
     */
    Intensity( std::string, int, std::string, double,
	       Point,Point, double, double, double, double );
    /*!
     *  \brief destructor
     *
     *  Destructo of the class Intensity
     *
     */
    ~Intensity(){/* Do nothing */};

  public:
    /*!
     *  \brief Operator =
     *
     *  Copy constructor of the class Intensity
     *
     */
    Intensity& operator =( const Intensity& );

  public:
    //
    // Physical properties
    //
    int    get_index_()const{return index_;};
    double get_I_()const{return I_;};
    double get_V_()const
    {
      if( type_of_potential_ ) return V_;
      else return get_electrical_potential();
    };
    void   set_V_( double V )
    {
      type_of_potential_ = true; // Punctual potential
      V_ = V;
    };
    //
    Point  get_r0_values_()const{return r0_values_;};
    double get_X_()const{return r0_values_.x();};
    double get_Y_()const{return r0_values_.y();};
    double get_Z_()const{return r0_values_.z();};
    //
    Point  get_r0_projection_()const{return r0_projection_;};
    std::size_t get_r0_projection_index_()const{return r0_projection_index_;};
    double get_projection_X_()const{return  r0_projection_.x();};
    double get_projection_Y_()const{return  r0_projection_.y();};
    double get_projection_Z_()const{return  r0_projection_.z();};
    //
    Point  get_e_values_()const{return e_values_;};
    double get_VX_()const{return e_values_.x();};
    double get_VY_()const{return e_values_.y();};
    double get_VZ_()const{return e_values_.z();};
    //
    std::string get_label_(){return label_;};
    //
    std::complex<double>  get_impedance_()const{return impedance_;};
    double get_Im_impedance_()const{return impedance_.imag();};
    double get_Re_impedance_()const{return impedance_.real();};
    //
    double get_surface_()const{return surface_;};
    double get_radius_()const{return radius_;};
    // 
    void   clear_electrical_potential_list(){electrical_potential_list_.clear();};
    double get_electrical_potential() const;
    
    //
    // Geometry propterties
    //
    void set_boundary_cells_(const std::map< std::size_t, std::list< MeshEntity  >  >& );
    //
    const std::map< std::size_t, std::list< MeshEntity  >  >& get_boundary_cells_()const{ return boundary_cells_;};
    const std::set<std::size_t>& get_boundary_vertices_()const{ return boundary_vertices_;};

  public:
    /*!
     *  \brief eval
     *
     *  This method returns the intensity at the electrode.
     *
     */
    double eval( const Point& , const ufc::cell& )const;
    /*!
     *  \brief Add potential
     *
     *  This method create the electrical potential mapping at the electrode contact surface.
     *
     */
    double add_potential_value( const double U )
    {
      type_of_potential_ = false; // Surface potential
      electrical_potential_list_.push_back( U );
    };
  };
  /*!
   *  \brief Dump values for Intensity
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Intensity& );
}
#endif
