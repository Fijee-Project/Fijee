#ifndef _Electrodes_INJECTION_H
#define _Electrodes_INJECTION_H
#include <dolfin.h>
#include <vector>
#include <string>
#include <algorithm>
//#include <iterator>
//
// UCSF
//
#include "Intensity.h"
//
//
//
using namespace dolfin;
/*!
 * \file Electrodes_injection.h
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
  /*! \class Electrodes_injection
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Electrodes_injection : public Expression
  {
    //! Electrodes list
    std::map< std::string, Solver::Intensity > electrodes_map_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_injection
     *
     */
  Electrodes_injection();
    /*!
     *  \brief Copy Constructor
     *
     *  Copy constructor of the class Electrodes_injection
     *
     */
    Electrodes_injection( const Electrodes_injection& );
    /*!
     *  \brief destructor
     *
     *  Destructo of the class Electrodes_injection
     *
     */
    ~Electrodes_injection(){/* Do nothing */};

  public:
    /*!
     *  \brief Operator =
     *
     *  Copy constructor of the class Electrodes_injection
     *
     */
    Electrodes_injection& operator =( const Electrodes_injection& );


  private:
    /*!
     *  \brief eval amplitude at one vertex
     *
     * A loop over all the facets of the mesh call eval function. The same Array of position can be called multiple time, because the same vertex belong to multiple facet.
     *
     */
    virtual void eval(Array<double>& , const Array<double>& , const ufc::cell& )const;
    /*!
     *  \brief value_rank
     *
     *  This method returns the rank of the tensor
     *  tensor 2: matrix
     *  tensor 1: vector
     *  tensor 0: scalar
     *
     */
    virtual std::size_t value_rank() const
      {
	return 0;
      }
    /*!
     *  \brief value_dimension
     *
     *  This method return the dimension
     *
     */
    virtual std::size_t value_dimension(uint i) const
      {
	return 3;
      }

  public:
    /*!
     *  \brief add_electrode
     *
     *  This method add a new electrode in the current injection system. 
     *
     */
    void add_electrode( std::string, int, std::string, double,
			Point, Point,
			double, double, double, double  );
    /*!
     *  \brief inside
     *
     *  This method check if a point is inside an electrode
     *
     */
    bool inside( const Point&  )const;
    /*!
     *  \brief inside
     *
     *  This method check if a point is inside an electrode
     *
     */
    std::tuple<std::string, bool> inside_probe( const Point&  )const;
    /*!
     *  \brief Set boundary cells
     *
     *  This method record the cell index per probes.
     *
     */
    void set_boundary_cells( const std::map< std::string, std::map< std::size_t, std::list< MeshEntity  >  >  >& );
    /*!
     *  \brief Set boundary vertices
     *
     *  This method record the vertices index per probes.
     *
     */
    void set_boundary_vertices( const std::map< std::string, std::set< std::size_t > >& );
  };
  /*!
   *  \brief Dump values for Electrodes_injection
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   */
  std::ostream& operator << ( std::ostream&, const Electrodes_injection& );
}
#endif
