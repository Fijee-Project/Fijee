#ifndef MINIMIZER_H
#define MINIMIZER_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Electrode.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
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
      int interation_;

    public:
      /*!
       *  \brief Constructor
       *
       *  Constructor of the class Minimizer
       */
      It_minimizer(){};
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
      It_minimizer( const It_minimizer& ){};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class It_minimizer_sphere
       *
       */
      It_minimizer& operator = ( const It_minimizer& ){return *this;};

    public:
      /*!
       *  \brief minimize function
       *
       *  This method launch the minimization algorithm
       */
      virtual void minimize()
      {
	std::cerr << "This is not a minimization algorithm. Look for daughter classes, "
		  << "e.g. Downhill simplex algorithm." 
		  << std::endl;
	abort();
      };
    };
  };
};
#endif
