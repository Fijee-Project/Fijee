#ifndef PDE_SOLVER_PARAMETERS_H_
#define PDE_SOLVER_PARAMETERS_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file PDE_solver_parameters.hh
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <stdlib.h>     /* getenv */
#include <string>
#include <errno.h>      /* builtin errno */
//
// Dolfin (FEniCS-project)
//
#include <dolfin.h>
//
// Get built-in type.  Creates member get_"name"() (e.g., get_visibility());
//
#define ucsf_get_macro(name,type) \
  inline type get_##name()const { \
    return this->name;		  \
  } 
//
// Get character string.  Creates member get_"name"() 
// (e.g., char *GetFilename());
//
#define ucsf_get_string_macro(name) \
  const char* get_##name() const {  \
    return this->name.c_str();	    \
  } 
//
//
//
/*! \namespace Domains
 * 
 * Name space for our new package
 * 
 *
 */
namespace Solver
{
  /*! \class PDE_solver_parameters
   *  \brief class representing whatever
   *
   *  This class provides for encapsulation of persistent state information. It also avoids the issue of which code segment should "own" the static persistent object instance. It further guarantees what mechanism is used to allocate memory for the underlying object and allows for better control over its destruction.
   */
  class PDE_solver_parameters
  {
  private:
    //! unique instance
    static PDE_solver_parameters* parameters_instance_;

    //
    // PDE solver parameters
    //! linear_solver
    std::string linear_solver_;
    //! Maximum iterations
    int maximum_iterations_;
    //! Relative tolerance
    double relative_tolerance_;
    //! Preconditioner
    std::string preconditioner_;

    //
    // Dispatching information
    //! Number of threads in the dispatching pool
    int number_of_threads_;

  protected:
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class PDE_solver_parameters
     *
     */
    PDE_solver_parameters();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    PDE_solver_parameters( const PDE_solver_parameters& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class PDE_solver_parameters
     */
    virtual ~PDE_solver_parameters();
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class PDE_solver_parameters
     *
     */
    PDE_solver_parameters& operator = ( const PDE_solver_parameters& );



  public:
    /*!
     *  \brief Get the singleton instance
     *
     *  This method return the pointer parameters_instance_
     *
     */
    static PDE_solver_parameters* get_instance();
    /*!
     *  \brief Get linear_solver_
     *
     *  This method return the linear solver selected
     *
     */
    ucsf_get_string_macro(linear_solver_);
    /*!
     *  \brief Get maximum_iterations_
     *
     *  This method return the maximum of iterations if the 
     * convergence is not reached
     *
     */
    ucsf_get_macro(maximum_iterations_, int);
    /*!
     *  \brief Get relative_tolerance_
     *
     *  This method return the relative tolerance
     *
     */
    ucsf_get_macro(relative_tolerance_, double);
    /*!
     *  \brief Get preconditioner_
     *
     *  This method return the preconditioner selected
     *
     */
    ucsf_get_string_macro(preconditioner_);
    /*!
     *  \brief Get number_of_threads_
     *
     *  This method return the number of threads dispatched
     *
     */
    ucsf_get_macro(number_of_threads_, int);
    /*!
     *  \brief Kill the singleton instance
     *
     *  This method kill the singleton parameters_instance pointer
     *
     */
    static void kill_instance();
    /*!
     *  \brief init parameters
     *
     *  This method initialize the simulation parameters.
     *
     */
    void init();

  };
  /*!
   *  \brief Dump values for PDE_solver_parameters
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const PDE_solver_parameters& );
}
#endif
