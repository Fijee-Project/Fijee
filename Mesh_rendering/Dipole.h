#ifndef DIPOLE_H_
#define DIPOLE_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Dipole.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
//
// UCSF
//
#include "Point_vector.h"
#include "Cell_conductivity.h"
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
  /*! \class Dipole
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   * Point_vector discribe the position of the dipole; the direction of the dipole. 
   * member weight_ of Point_vector class represent the intensity (mA) of the dipole.
   */
  class Dipole : public Domains::Point_vector
  {
  private:
    //! Cell number
    int cell_id_;
    //! Cell domain 
    int cell_subdomain_;
    //! Conductivity coefficients
    //! 0 -> C00
    //! 1 -> C01
    //! 2 -> C02
    //! 3 -> C11
    //! 4 -> C12
    //! 5 -> C22
    float conductivity_coefficients_[6];
    //! Conductivity eigenvalues
    float conductivity_eigenvalues_[3];

    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Dipole
     *
     */
    Dipole();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Dipole( const Dipole& );
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor from Cell_conductivity object
     *
     */
    Dipole( const Cell_conductivity& );
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Dipole
     */
    virtual ~Dipole();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Dipole
     *
     */
    Dipole& operator = ( const Dipole& );
    
  public:
    int get_cell_id_() const {return cell_id_;};
    int get_cell_subdomain_() const {return cell_subdomain_;};

    const float* get_conductivity_coefficients_() const {return conductivity_coefficients_; }

    float C00() const { return conductivity_coefficients_[0]; }
    float C01() const { return conductivity_coefficients_[1]; }
    float C02() const { return conductivity_coefficients_[2]; }
    float C11() const { return conductivity_coefficients_[3]; }
    float C12() const { return conductivity_coefficients_[4]; }
    float C22() const { return conductivity_coefficients_[5]; }

    float& C00() { return conductivity_coefficients_[0]; }
    float& C01() { return conductivity_coefficients_[1]; }
    float& C02() { return conductivity_coefficients_[2]; }
    float& C11() { return conductivity_coefficients_[3]; }
    float& C12() { return conductivity_coefficients_[4]; }
    float& C22() { return conductivity_coefficients_[5]; }

    float lambda1() const { return conductivity_eigenvalues_[0]; }
    float lambda2() const { return conductivity_eigenvalues_[1]; }
    float lambda3() const { return conductivity_eigenvalues_[2]; }

    float& lambda1() { return conductivity_eigenvalues_[0]; }
    float& lambda2() { return conductivity_eigenvalues_[1]; }
    float& lambda3() { return conductivity_eigenvalues_[2]; }
 };
  /*!
   *  \brief Dump values for Dipole
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Dipole& );
};
#endif
