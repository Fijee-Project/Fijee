#ifndef LABELED_DOMAIN_H_
#define LABELED_DOMAIN_H_
#include <iostream>
#include <string>
#include <fstream>
//
// UCSF
//
#include "Utils/enum.h"
//
// Project
//
//
//
//
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Labeled_domain.h
 * \brief brief explaination 
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
  /*! \class Labeled_domain
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  // Implicite_domain = 
  template< typename Implicite_domain, typename Point_type, typename VectorPointNormal>
  class Labeled_domain
  {
  private:
    Implicite_domain* implicite_domain_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Labeled_domain
     *
     */
  Labeled_domain():
    implicite_domain_( nullptr )
      {};
    /*!
     *  \brief Constructor
     *
     *  Constructor of the class Labeled_domain
     *
     */
    Labeled_domain( const char* File ):
    implicite_domain_( new Implicite_domain( File ) )
      {};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Labeled_domain( const Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >& )
      {};
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Labeled_domain( Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >&& )
      {};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Labeled_domain
     */
    virtual ~Labeled_domain()
      {
	delete implicite_domain_;
	implicite_domain_ = NULL;
      };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Labeled_domain
     *
     */
    Labeled_domain& operator = ( const Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >& )
      {};
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Labeled_domain
     *
     */
    Labeled_domain& operator = ( Labeled_domain< Implicite_domain, Point_type, VectorPointNormal >&& )
      {};
    /*!
     *  \brief Move Operator ()
     *
     *  Move operator of the class Labeled_domain
     *
     */
    void operator ()( double** Positions )
    {
      (*implicite_domain_)( Positions );
    };

  public:
    /*!
     *  \brief Get max_x_ value
     *
     *  This method return the maximum x coordinate max_x_.
     *
     *  \return max_x_
     */
    //    inline double get_max_x( ) const {return max_x_;};
 
  public:
    /*!
     *  \brief Inside domain
     *
     *  This method check if a point is inside the implicite domain
     *
     */
    inline bool inside_domain( Point_type Point_Type )
    {
      return implicite_domain_->inside_domain( Point_Type );
    };
    /*!
     *  \brief
     */
    inline const double* get_poly_data_bounds_()
    {
      return implicite_domain_->get_poly_data_bounds_();
    };
    /*!
     *  \brief Get point_normal vector
     *
     *  This method return point_normal_ of the STL mesh.
     *
     *  \return point_normal_
     */
    inline VectorPointNormal get_point_normal()
    {
      return implicite_domain_->get_point_normal_();
    };
  };
  /*!
   *  \brief Dump values for Labeled_domain
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
//  std::ostream& operator << ( std::ostream&, const Labeled_domain< Implicite_domain >& );
};
#endif
