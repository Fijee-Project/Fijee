#ifndef DOWNHILL_SIMPLEX_H
#define DOWNHILL_SIMPLEX_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Downhill_simplex_sphere.h
 * \brief brief describe 
 * \author John Zheng He, Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <vector>
//
// UCSF
//
#include "Minimizer.h"
//
//
//
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \namespace Minimizers
   * 
   * Name space for our new package
   *
   */
  namespace Minimizers
  {
    /*! \class Shape
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class Downhill_simplex : public It_minimizer
    {
    private:
      //! Simplex vertices
      std::vector< Estimation_tuple > simplex_;
      //! Map of conductivity boundary values for each tissues
      std::vector< std::tuple<double, double> > conductivity_boundaries_;
      //! Funtion to minimize
      Function function_;
      //! Tolerance
      double delta_;
      //! TODO
      double a_;
      //! TODO
      double b_;
      //! TODO
      double c_;
      //! TODO
      std::vector< int > asc_;

    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Downhill_simplex_sphere
       *
       */
      Downhill_simplex();
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Downhill_simplex_sphere
       */
      virtual ~Downhill_simplex(){/* Do nothing*/};
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
    Downhill_simplex( const Downhill_simplex& that):It_minimizer(that){};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Downhill_simplex_sphere
       *
       */
      Downhill_simplex& operator = ( const Downhill_simplex& that)
	{
	  It_minimizer::operator=(that);
	  return *this;
	};

      
    public:
      /*!
       *  \brief minimize function
       *
       *  This method initialize the minimizer
       */
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& );
      /*!
       *  \brief minimize function
       *
       *  This method launch the minimization algorithm
       */
      virtual void minimize();

    private:
      /*!
       *  \brief Order the simplex vertices
       *
       *  This method order the simplex vertices
       */
      void order_vertices();
      /*!
       *  \brief get facet centroid
       *
       *  This method 
       */
      const Eigen::Vector3d get_facet_centroid( const Eigen::Vector3d&, 
						const Eigen::Vector3d&, 
						const Eigen::Vector3d& ) const;
      /*!
       *  \brief Convergence criteria
       *
       *  This method 
       */
      bool is_converged();
      /*!
       *  \brief Contraction
       *
       *  This method TODO
       */
      void contraction();
      /*!
       *  \brief Transform
       *
       *  This method TODO
       */
      void transform(){};
      /*!
       *  \brief Reflection
       *
       *  This method TODO
       */
      Eigen::Vector3d reflection();
      /*!
       *  \brief Get middle
       *
       *  This method TODO
       */
      const Eigen::Vector3d get_middle( const Eigen::Vector3d&, 
					const Eigen::Vector3d& ) const;
    };
    /*!
     *  \brief Dump values for Electrode
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     */
    std::ostream& operator << ( std::ostream&, const Downhill_simplex& );
  }
}
#endif
