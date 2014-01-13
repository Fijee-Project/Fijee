#ifndef _MESH_GENERATOR_H
#define _MESH_GENERATOR_H
#include <iostream>
#include <thread>
#include <string>
#include <list>
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Access_parameters.h"
#include "Utils/enum.h"
#include "Cell_conductivity.h"
#include "Build_dipoles_list.h"
#include "Build_dipoles_list_knn.h"
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
  template < typename Labeled_domain, typename Conductivity, typename Mesher >
  class Mesh_generator
  {
  private:
    Labeled_domain domains_;
    Conductivity   tensors_;
    Mesher         mesher_;
    //! List of cell with matching conductivity coefficients
    std::list< Cell_conductivity > list_cell_conductivity_;

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
      // Tetrahedrization
      mesher_.Tetrahedrization();

      //
      // Diffusion tensor
      tensors_.make_conductivity( mesher_.get_mesh_() );
    }
    /*!
     *  \brief 
     *
     *  
     *
     */
    void make_output()
    {
      //
      // Algorithm building the dipoles list;
      Build_dipoles_list_knn dipoles_;
      dipoles_.Make_list( tensors_.get_list_cell_conductivity_() );

#ifdef DEBUG
      // DEBUG MODE
      // Sequencing
      mesher_.Output_mesh_format();
      mesher_.Output_FEniCS_xml();
      tensors_.Output_mesh_conductivity_xml();
      dipoles_.Output_dipoles_list_xml();
      //#ifdef TRACE
      //#if ( TRACE == 200 )
      //  mesher_.Output_VTU_xml();
      //#endif
      //#endif
      //
#else
  // NO DEBUG MODE
  // Multi-threading
      std::thread output(std::ref(mesher_), MESH_OUTPUT);
      std::thread subdomains(std::ref(mesher_), MESH_SUBDOMAINS);
      std::thread conductivity(std::ref(tensors_));
      std::thread dipoles(std::ref(dipoles_));
      //
      //#ifdef TRACE
      //#if ( TRACE == 200 )
      //  std::thread vtu(std::ref(mesher_), MESH_VTU);
      //  vtu.join();
      //#endif
      //#endif
      //
      output.join();
      subdomains.join();
      conductivity.join();
      dipoles.join();
#endif
    }
  };
}
#endif
