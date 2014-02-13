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
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
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
    //! Electrode direction vector
    Point e_values_;  
    //! Electrode intensity [I_] = A
    double I_;
    //! Electrode name
    std::string label_;
    //! Electrode impedance
    std::complex<double> impedance_;
    //! Contact suface between the electrode and the scalp
    double surface_;
    //! Contact suface radius between the electrode and the scalp
    double radius_;

    //
    // Geometry propterties
    //
    //! Set of boundaries mesh cells
    mutable std::set<std::size_t> boundary_cells_;
    //!
    mutable bool not_yet_;
    

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
    //
    Point  get_r0_values_()const{return r0_values_;};
    double get_X_()const{return r0_values_.x();};
    double get_Y_()const{return r0_values_.y();};
    double get_Z_()const{return r0_values_.z();};
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
    // Geometry propterties
    //
    void set_boundary_cells_(const std::set<std::size_t>& Boundary_cells)const{ boundary_cells_ = Boundary_cells;};
    const std::set<std::size_t>& get_boundary_cells_()const{ return boundary_cells_;};

  public:
    /*!
     *  \brief eval
     *
     *  This method returns the intensity at the electrode.
     *
     */
    double eval( const Point& , const ufc::cell& )const;
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
