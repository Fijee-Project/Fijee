#ifndef CONDUCTIVITY_TENSOR_H_
#define CONDUCTIVITY_TENSOR_H_
//
// UCSF
//
#include "Utils/Statistical_analysis.h"
#include "Utils/Fijee_environment.h"
#include "Utils/enum.h"
#include "CGAL_tools.h"
/*!
 * \file Conductivity_tensor.h
 * \brief brief describe 
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
  /*! \class Conductivity_tensor
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Conductivity_tensor : public Utils::Statistical_analysis
  {
  public:
    virtual ~Conductivity_tensor(){/* Do nothing */};

    /*!
     *  \brief Operator ()
     *
     *  Object function for multi-threading
     *
     */
    virtual void operator ()() = 0;
//
//  private:
//    /*!
//     *  \brief Move_conductivity_array_to_parameters
//     *
//     *  This method moves members array to Access_Parameters's object.
//     *
//     */
//    void move_conductivity_array_to_parameters();
//
  public:
    /*!
     */
    virtual void Make_analysis() = 0;
    /*!
     *  \brief Move_conductivity_array_to_parameters
     *
     *  This method moves members array to Access_Parameters's object.
     *
     */
    virtual void make_conductivity( const C3t3& ) = 0;
    /*!
     *  \brief Output the XML match between mesh and conductivity
     *
     *  This method matches a conductivity tensor for each cell.
     *
     */
    virtual void Output_mesh_conductivity_xml() = 0;
//    /*!
//     *  \brief VTK visualization
//     *
//     *  This method gives a screenshot of the brain diffusion/conductivity vector field.
//     *
//     */
//    void VTK_visualization();
//    /*!
//     *  \brief 
//     *
//     *  This method 
//     *
//     */
//    void INRIMAGE_image_of_conductivity_anisotropy();
  };
};
#endif
