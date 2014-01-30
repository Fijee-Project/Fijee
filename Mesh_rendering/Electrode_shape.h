#ifndef ELECTRODE_SHAPE_H_
#define ELECTRODE_SHAPE_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Electrode.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
//
// UCSF
//
#include "Point.h"
#include "Point_vector.h"
#include "Access_parameters.h"
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

  /*! \class Electrode shape
   * \brief classe representing whatever
   *
   *  This class is an example of class 
   * 
   */
  class Electrode_shape
  {
  public:
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Electrode_shape_sphere
     */
    virtual ~Electrode_shape(){/* Do nothing */};  

  public:
    /*!
     *  \brief inside 
     *
     *  This function check if the point (X, Y, Z) is inside the shape.
     *
     */
    virtual bool inside( Domains::Point_vector&, 
			 float, float, float ) = 0;
    /*!
     *  \brief Size of the contact surface
     *
     *  This function return the size of the contect surface between the electrode and the scalp.
     *
     */
    virtual float contact_surface( ) = 0;
  };
};
#endif
