#ifndef _MESH_GENERATOR_H
#define _MESH_GENERATOR_H
#include <iostream>
#include <thread>
#include <string>
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Access_parameters.h"
//
//
//
//////typedef Solver::PDE_solver_parameters SDEsp;
//
//
//
//using namespace dolfin;
//
/*!
 * \file Mesh_generator.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Mesh_generator
   * \brief classe representing the dipoles distribution
   *
   *  This class is an example of class I will have to use
   */
  template < typename Labeled_domain, typename Conductivity >
  class Mesh_generator
  {
  private:
    Labeled_domain domains_;
    Conductivity   tensors_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Mesh_generator
     *
     */
  Mesh_generator()
    {};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Mesh_generator( const Mesh_generator& ){};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Mesh_generator
     *
     */
    Mesh_generator& operator = ( const Mesh_generator& ){};
//    /*!
//     *  \brief Operator ()
//     *
//     *  Operator () of the class Mesh_generator
//     *
//     */
//    void operator () ();

  public:
    /*!
     *  \brief construct
     *
     *  
     *
     */
    void make_inrimage()
    {
      //
      // segmentation: create the segmentation from outputs
      domains_.model_segmentation();

      //
      // Write Inrimage
      domains_.write_inrimage_file();
    }
    /*!
     *  \brief 
     *
     *  
     *
     */
    void make_conductivity()
    {
      //
      // Diffusion tensor
      tensors_.make_conductivity();
    }
  };
}
#endif
