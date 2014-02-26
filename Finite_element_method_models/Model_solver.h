#ifndef _MODEL_SOLVER_H
#define _MODEL_SOLVER_H
#include <list>
#include <memory>
#include <string>
//
// FEniCS
//
#include <dolfin.h>
// Source localization subtraction model
#include "SLS_model.h"
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "PDE_solver_parameters.h"
#include "Utils/Thread_dispatching.h"
//
//
//
typedef Solver::PDE_solver_parameters SDEsp;
//
//
//
//using namespace dolfin;
//
/*!
 * \file Model_solver.h
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
  /*! \class Model_solver
   * \brief classe representing the dipoles distribution
   *
   *  This class is an example of class I will have to use
   */
  template < typename Physical_model, int num_of_threads = 1 >
  class Model_solver
  {
  private:
    Physical_model model_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Model_solver
     *
     */
  Model_solver()
    {};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Model_solver( const Model_solver& ){};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Model_solver
     *
     */
    Model_solver& operator = ( const Model_solver& ){};
//    /*!
//     *  \brief Operator ()
//     *
//     *  Operator () of the class Model_solver
//     *
//     */
//    void operator () ();

  public:
    /*!
     */
    void solver_loop()
    {
      
      //
      // Define the number of threads in the pool of threads
      Utils::Thread_dispatching pool( num_of_threads );
      
      
      //
      //
      int tempo = 0;
      for( int physical_event = 0 ;
	   physical_event != model_.get_number_of_physical_event() ; 
	   physical_event++ )
	if( ++tempo < 10 )
	  {
	    //
	    // Enqueue tasks
	    pool.enqueue( std::ref(model_) );
	  }
	else {break;}
    };
  };
}
#endif
