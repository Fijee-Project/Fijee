#ifndef ELECTRODE_H_
#define ELECTRODE_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Electrode.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
#include <complex>
//
// UCSF
//
#include "Utils/enum.h"
#include "Point.h"
#include "Point_vector.h"
#include "Access_parameters.h"
#include "Electrode_shape.h"
#include "Electrode_shape_sphere.h"
#include "Electrode_shape_cylinder.h"
//
// Eigen
//
#include <Eigen/Dense>
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Electrode
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   * Point_vector discribe the position of the dipole; the direction of the dipole. 
   * member weight_ of Point_vector class represent the intensity (mA) of the dipole.
   */
  class Electrode : public Domains::Point_vector
  {
  private:
    //! Index of the electrode
    int index_;
    //! Label of the electrode
    std::string label_;
    //! Electrode intensity
    float intensity_;
    //! Electrode impedance
    std::complex<float> impedance_;
    //! Electrode shape
    std::shared_ptr< Domains::Electrode_shape > shape_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrode
     *
     */
    Electrode();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrode
     *
     */
    Electrode( int, std::string, 
	       float, float, float,
	       float, float, float );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Electrode
     */
    virtual ~Electrode();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Electrode( const Electrode& );
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Electrode
     *
     */
    Electrode& operator = ( const Electrode& );
    
  public:
    ucsf_get_macro(index_, int);
    ucsf_get_macro(label_, std::string);
    ucsf_get_macro(intensity_, float);
    ucsf_get_macro(impedance_, std::complex<float>);
    ucsf_get_macro(shape_, std::shared_ptr< Domains::Electrode_shape > );

  public:
    /*!
     *  \brief inside electrode domain
     *
     *  This function check if the point (X, Y, Z) is inside the electrode.
     *
     *  \param X : x-position of the checl point
     *  \param Y : y-position of the checl point
     *  \param Z : z-position of the checl point
     *
     */
    bool inside_domain(float X, float Y, float Z)
    {
      return shape_->inside( *this /*Center*/, X, Y, Z );
    }
 };
  /*!
   *  \brief Dump values for Electrode
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Electrode& );
};
#endif
