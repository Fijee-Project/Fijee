#ifndef MINIMIZER_H
#define MINIMIZER_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Minimization.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <tuple>
#include <functional>
//
// Eigen
//
#include <Eigen/Dense>
//
// UCSF
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
    typedef std::function< double( const Eigen::Vector3d& ) > Function;
    typedef std::tuple< 
      double,          /* - 0 - estimation */
      Eigen::Vector3d /* - 1 - sigma (0) skin, (1) skull compacta, (2) skull spongiosa */
      > Estimation_tuple;
    /*! \class Minimizer
     * \brief classe representing whatever
     *
     *  This class is an example of class 
     * 
     */
    class Minimizer
    {
    public:
      virtual ~Minimizer(){/* Do nothing */};  
      
    public:
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& ) = 0;
      virtual void minimize() = 0;
    };
    /*! \class It_minimizer
     * \brief classe representing the 
     *
     *  This class is an example of class I will have to use
     */
    class It_minimizer : public Minimizer
    {
    protected:
      //! Number of iteration
      int iteration_;
      //! Max number of iterations
      int max_iterations_;

    public:
      /*!
       *  \brief Constructor
       *
       *  Constructor of the class Minimizer
       */
    It_minimizer():iteration_(0), max_iterations_(200){};
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Minimizer
       */
      virtual ~It_minimizer(){/* Do nothing */};
            /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
    It_minimizer( const It_minimizer& that):
      iteration_(that.iteration_), max_iterations_(that.max_iterations_){};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class It_minimizer_sphere
       *
       */
      It_minimizer& operator = ( const It_minimizer& that )
	{
	  iteration_      = that.iteration_;
	  max_iterations_ = that.max_iterations_;
	  //
	  return *this;
	};

    public:
      /*!
       *  \brief initialization function
       *
       *  This method initialize the minimizer
       */
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& ) = 0;
      /*!
       *  \brief minimize function
       *
       *  This method launch the minimization algorithm
       */
      virtual void minimize() = 0;
    };
  }
}
#endif
