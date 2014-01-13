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
#include "Utils/Fijee_environment.h"
#include "Conductivity_tensor.h"
#include "Cell_conductivity.h"
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
  private:
    //! List of cell with matching conductivity coefficients
    std::list< Cell_conductivity > list_cell_conductivity_;


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
    virtual void operator ()()
    {
      Output_mesh_conductivity_xml();
    };

  private:
    /*!
     */
    virtual void Make_analysis(){};

  public:
    /*!
     *  \brief Move_conductivity_array_to_parameters
     *
     *  This method moves members array to Access_Parameters's object.
     *
     */
    virtual void make_conductivity( const C3t3& ){};
    /*!
     *  \brief Output the XML match between mesh and conductivity
     *
     *  This method matches a conductivity tensor for each cell.
     *
     */
    virtual void Output_mesh_conductivity_xml(){};
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
