#ifndef _SOURCES_H
#define _SOURCES_H
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Source.h"
//
//
//
//using namespace dolfin;
//
/*!
 * \file Sources.h
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
  /*! \class Sources
   * \brief classe representing the dipoles distribution
   *
   *  This class is an example of class I will have to use
   */
  class Sources
  {
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Sources
     *
     */
    Sources();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Sources( const Sources& ) = delete;
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Sources( Sources&& ) = delete;
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Sources
     */
    virtual ~Sources(){};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Sources
     *
     */
    Sources& operator = ( const Sources& ) = delete;
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Sources
     *
     */
    Sources& operator = ( Sources&& ) = delete;

  private:
    /// Number of dipoles
    int number_dipoles_;

  };
}
#endif
