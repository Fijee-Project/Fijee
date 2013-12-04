#ifndef _SOURCE_H
#define _SOURCE_H
#include <dolfin.h>
#include <vector>
#include <string>

using namespace dolfin;
/*!
 * \file Source.h
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
  /*! \class Phi
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Phi : public Expression
  {
  private:
    //! Dipole index
    int index_;
    //! Dipole position
    std::vector<double> r0_values_;  
    //! Dipole direction
    std::vector<double> e_values_;  
    //! Dipole intensity [Q_] = A.m
    double Q_;
    //! homogeneous conductivity eigenvalues - [a0_] = S/m
    double a0_;
    //! Dipole name
    std::string name_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Phi
     *
     */
    Phi();
    /*!
     *  \brief Copy Constructor
     *
     *  Copy constructor of the class Phi
     *
     */
    Phi( const Phi& );
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Phi
     *
     */
    Phi( int, double,
	 double, double, double, 
	 double, double, double,
	 double, double, double );
    /*!
     *  \brief destructor
     *
     *  Destructo of the class Phi
     *
     */
    ~Phi(){/* Do nothing */};

  public:
    /*!
     *  \brief Operator =
     *
     *  Copy constructor of the class Phi
     *
     */
    Phi& operator =( const Phi& );

  public:
    int    get_index_()const{return index_;};
    double get_Q_()const{return Q_;};
    double get_a0_()const{return a0_;};

    double get_X_()const{return r0_values_[0];};
    double get_Y_()const{return r0_values_[1];};
    double get_Z_()const{return r0_values_[2];};
    double get_VX_()const{return e_values_[0];};
    double get_VY_()const{return e_values_[1];};
    double get_VZ_()const{return e_values_[2];};

    std::string get_name_(){return name_;};

  private:
    /*!
     */
    void eval(Array<double>& values, const Array<double>& x) const;
  };
  /*!
   *  \brief Dump values for Phi
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Phi& );
}
#endif
