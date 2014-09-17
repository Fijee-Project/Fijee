//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#ifndef _Electrodes_INJECTION_H
#define _Electrodes_INJECTION_H
#include <dolfin.h>
#include <vector>
#include <string>
#include <algorithm>
//#include <iterator>
//
// UCSF project
//
#include "Utils/Fijee_environment.h"
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
   * \brief classe representing a set of electrodes
   *
   *  This class implements a set of electrode. Typically, it is associated with a type of electroencephalogram.
   */
  class Electrodes_injection : public Expression
  {
    //! Electrodes list
    std::map< /* label */ std::string, 
      /* electrode caracteristics */ Solver::Intensity > electrodes_map_;
    //! Electrodes list
    std::map< /* label */ std::string, 
      /* electrical potential */ double > potential_measured_map_;
    //! Time injection/measure
    double time_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_injection
     *
     */
  Electrodes_injection();
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Electrodes_injection
     *
     */
  Electrodes_injection(double);
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

  public:
    /*!
     *  \brief value_dimension
     *
     *  This method return the dimension
     *
     */
    std::map< std::string, Solver::Intensity > get_electrodes_map_()const{return electrodes_map_;};

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
     *  \brief Get the sampling time
     *
     *  This method return the time_ member.
     *
     */
    ucsf_get_macro( time_, double );

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
     *  \brief add_measured_potential
     *
     *  This method add the measured potential at each electrode. 
     *  This functionnality is used for the conductivity estimation.
     *
     */
    void add_measured_potential( std::string, double, double  );
    /*!
     *  \brief inside
     *
     *  This method check if a point is inside an electrode
     *
     */
    bool inside( const Point&  )const;
    /*!
     *  \brief Add potential value
     *
     *  This method check if a point is inside an electrode and add the potential value to the electrode list of potential
     *
     */
    bool add_potential_value( const Point&, const double );
    /*!
     *  \brief 
     *
     *  This method Add potential value
     *
     */
    bool add_potential_value( const std::string, const double );
    /*!
     *  \brief Inside electrode probe
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
     *  \brief Punctual potential evaluation
     *
     *  This method 
     *
     *  \param U: function solution of the Partial Differential Equation.
     *  \param Mesh: Tetrahedrization mesh
     *
     */
    void punctual_potential_evaluation( const dolfin::Function&, 
					const std::shared_ptr< const Mesh >  );
     /*!
     *  \brief Punctual potential evaluation
     *
     *  This method 
     *
     *  \param U: function solution of the Partial Differential Equation.
     *  \param Mesh: Tetrahedrization mesh
     *
     */
    void surface_potential_evaluation( const dolfin::Function&, 
				       const std::shared_ptr< const Mesh >  );
   /*!
     *  \brief Information
     *
     *  This method retrieves infomation for a specific electrode.
     *
     *  \param label: labe of the specific electrode
     *
     */
    const Solver::Intensity& information( const std::string label ) const
      {
	//
	//
	auto electrode = electrodes_map_.find(label);
	//
	if( electrode !=  electrodes_map_.end() )
	  return electrode->second;
	else
	  {
	    std::cerr << "Error: electrode " 
		      << label 
		      << " does not belong the ste of electrodes!" << std::endl;
	    abort();
	  }
      };
      /*!
     *  \brief Sum of squares
     *
     *  This method process the sum of squares formula between the measured potential and the simulated potential:
     * S(\beta) = (U - \phi(\beta))^{T} \Sigma^{-1} (U - \phi(\beta))
     *
     */
    double sum_of_squares() const;
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
