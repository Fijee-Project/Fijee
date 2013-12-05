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


  /*! \class Current_density
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Current_density : public Expression
  {
  private:
    //! Dipole index
    int index_;
    //! Cell index where the dipole is located
    int index_cell_;
    //! Dipole position
    std::vector<double> r0_values_;  
    //! Dipole direction
    std::vector<double> e_values_;  
    //! Dipole intensity [Q_] = A.m
    double Q_;
    //! Dipole name
    std::string name_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Current_density
     *
     */
    Current_density();
    /*!
     *  \brief Copy Constructor
     *
     *  Copy constructor of the class Current_density
     *
     */
    Current_density( const Current_density& );
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Current_density
     *
     */
    Current_density( int, int, double,
		     double, double, double, 
		     double, double, double );
    /*!
     *  \brief destructor
     *
     *  Destructo of the class Current_density
     *
     */
    ~Current_density(){/* Do nothing */};

  public:
    /*!
     *  \brief Operator =
     *
     *  Copy constructor of the class Current_density
     *
     */
    Current_density& operator =( const Current_density& );

  public:
    int    get_index_()const{return index_;};
    int    get_index_cell_()const{return index_cell_;};
    double get_Q_()const{return Q_;};

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
    void eval(Array<double>& , const Array<double>& , const ufc::cell& ) const;
     /*!
     *  \brief value_rank
     *
     *  This method returns the rank of the tensor
     *
     */
    virtual std::size_t value_rank() const
      {
	return 1;
      }
    /*!
     *  \brief value_dimension
     *
     *  This method evaluates 
     *
     */
    virtual std::size_t value_dimension(uint i) const
      {
	return 3;
      }
 };
  /*!
   *  \brief Dump values for Current_density
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Current_density& );


}
#endif
