#ifndef BUILD_DIPOLES_LIST_H_
#define BUILD_DIPOLES_LIST_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Build_dipoles_list.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
// UCSF
//
#include "Utils/enum.h"
#include "Point_vector.h"
#include "Cell_conductivity.h"
#include "Utils/Statistical_analysis.h"
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Build_dipoles_list
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Build_dipoles_list : public Utils::Statistical_analysis
  {
  private:

    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Build_dipoles_list
     *
     */
    Build_dipoles_list();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Build_dipoles_list( const Build_dipoles_list& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Build_dipoles_list
     */
    virtual ~Build_dipoles_list();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Build_dipoles_list
     *
     */
    Build_dipoles_list& operator = ( const Build_dipoles_list& );
    
  public:
    /*!
     */
    virtual void Make_list( const std::list< Cell_conductivity >& List_cell_conductivity ) = 0;
    /*!
     */
    virtual void Build_stream(std::ofstream&) = 0;

  private:
    /*!
     */
    virtual void Make_analysis() = 0;
};
  /*!
   *  \brief Dump values for Build_dipoles_list
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Build_dipoles_list& );
};
#endif
