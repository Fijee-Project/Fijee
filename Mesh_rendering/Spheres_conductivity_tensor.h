#ifndef SPHERES_CONDUCTIVITY_TENSOR_H_
#define SPHERES_CONDUCTIVITY_TENSOR_H_
/*!
 * \file Spheres_conductivity_tensor.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
//
// UCSF
//
#include "Conductivity_tensor.h"
//
// Eigen
//
#include <Eigen/Dense>
// VTK
#include <vtkSmartPointer.h>
#include <vtkMatrix3x3.h>
//#include <vtkMath.h>

/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Spheres_conductivity_tensor
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Spheres_conductivity_tensor : public Conductivity_tensor
  {
 
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Spheres_conductivity_tensor
     *
     */
    Spheres_conductivity_tensor();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Spheres_conductivity_tensor( const Spheres_conductivity_tensor& );
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Spheres_conductivity_tensor( Spheres_conductivity_tensor&& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Spheres_conductivity_tensor
     */
    virtual ~Spheres_conductivity_tensor(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Spheres_conductivity_tensor
     *
     */
    Spheres_conductivity_tensor& operator = ( const Spheres_conductivity_tensor& );
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Spheres_conductivity_tensor
     *
     */
    Spheres_conductivity_tensor& operator = ( Spheres_conductivity_tensor&& );
    /*!
     *  \brief Move Operator ()
     *
     *  Object function for multi-threading
     *
     */
    void operator ()();

  private:
    /*!
     *  \brief Move_conductivity_array_to_parameters
     *
     *  This method moves members array to Access_Parameters's object.
     *
     */
    void move_conductivity_array_to_parameters();

  public:
    /*!
     *  \brief Move_conductivity_array_to_parameters
     *
     *  This method moves members array to Access_Parameters's object.
     *
     */
    void make_conductivity()
    {
      move_conductivity_array_to_parameters();
    };
    /*!
     *  \brief VTK visualization
     *
     *  This method gives a screenshot of the brain diffusion/conductivity vector field.
     *
     */
    void VTK_visualization();
  };
  /*!
   *  \brief Dump values for Spheres_conductivity_tensor
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Spheres_conductivity_tensor& );
};
#endif
