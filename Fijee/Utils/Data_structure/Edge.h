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
#ifndef EDGE_H
#define EDGE_H
/*!
 * \file Edge.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
#include <list>
//
// UCSF
//
#include "Fijee/Fijee_exception_handler.h"
#include "Utils/Data_structure/Vertex.h"
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Data_structure
  {
    /*! \class Edge
     * \brief classe representing an edge 
     *
     *  This class is a basic data structure for edges. In out case, it is use for undirected graphs, but a weight member is present for directed graph.
     */
    template< typename Type_point >
      class Edge
      {
      private:
	//! weight of the point. This member offert scalar information for the point. 
	Type_point weight_;
	//! vertices of the point.
	Vertex<Type_point> vertices_[2];


      public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Edge
	 *
	 */
	Edge();
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Edge
	 *
	 */
	Edge( const Vertex<Type_point> &, const Vertex<Type_point> &, 
	      Type_point Weight = static_cast<Type_point>(0.) );
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Edge( const Edge& );
	/*!
	 *  \brief Move Constructor
	 *
	 *  Constructor is a moving constructor
	 *
	 */
	Edge( Edge&& );
	/*!
	 *  \brief Destructeur
	 *
	 *  Destructor of the class Edge
	 */
	virtual ~Edge(){/*Do nothing*/};
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Edge
	 *
	 */
	Edge& operator = ( const Edge& );
	/*!
	 *  \brief Move Operator =
	 *
	 *  Move operator of the class Edge
	 *
	 */
	Edge& operator = ( Edge&& );

      public:
	const Vertex<Type_point>* get_vertices_() const {return vertices_; }

	void set_weight_( Type_point Weight ) {weight_ = Weight; }
	Type_point weight() const { return weight_; }
	Vertex<Type_point> V1() const { return vertices_[0]; }
	Vertex<Type_point> V2() const { return vertices_[1]; }

	Vertex<Type_point>& V1() { return vertices_[0]; }
	Vertex<Type_point>& V2() { return vertices_[1]; }

      public:
	/*!
	 *  \brief End vertices
	 *
	 *  This method returns vertex list containing edge's end vertices.
	 *
	 */
	std::list< Vertex<Type_point> >& end_vertices() const {};
	/*!
	 *  \brief Opposite vertex
	 *
	 *  This method returns the end vertex of edge distinct from vertex V
	 *
	 */
	const Vertex<Type_point>& opposite_vertex( const Vertex<Type_point>& V) const 
	  {
	    return ( vertices_[0] == V?vertices_[1]:vertices_[0] );
	  };
	/*!
	 *  \brief Is adjacent to
	 *
	 *  This method tests whether this edge and f are adjacent.
	 *
	 */
	bool is_adjacent_to( const Edge<Type_point>& ) const {};
	/*!
	 *  \brief Is incident on
	 *
	 *  This method tests whether this edge is incident on vertex V.
	 *
	 */
	bool is_incident_on( const Vertex<Type_point>& V ) const 
	{
	  return ( V == vertices_[0] || V == vertices_[1] );
	};
      };
    //
    //
    //
    template< typename Type_point >
      Edge<Type_point>::Edge(): weight_( static_cast<Type_point>(0.) )
      {
      }
    //
    //
    //
    template< typename Type_point >
      Edge<Type_point>::Edge( const Vertex<Type_point>& V1, 
			      const Vertex<Type_point>& V2, 
			      Type_point Weight ):
    weight_(Weight)
    {
      try{
	// 
	// 
	vertices_[0] = V1;
	vertices_[1] = V2;
	// 
	if( V1 == V2 )
	  {
	    std::string message = std::string("Two vertices of an Edge must be different!");
	    //
	    throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	  }
      }
      catch( Fijee::Exception_handler& err )
	{
	  std::cerr << err.what() << std::endl;
	}
    }
    //
    //
    //
    template< typename Type_point >
      Edge<Type_point>::Edge( const Edge<Type_point>& that ): 
    weight_(that.weight_)
      {
	vertices_[0] = that.vertices_[0];
	vertices_[1] = that.vertices_[1];
      }
    //
    //
    //
    template< typename Type_point >
      Edge<Type_point>::Edge( Edge<Type_point>&& that ): 
    weight_( static_cast<Type_point>(0.) )
      {
	// 
	// 
	weight_      = that.weight_;
	vertices_[0] = std::move( that.vertices_[0] );
	vertices_[1] = std::move( that.vertices_[1] );
	// 
	that.weight_      = static_cast<Type_point>(0.);	
      }
    //
    //
    //
    template< typename Type_point > Edge<Type_point>& 
      Edge<Type_point>::operator = ( const Edge<Type_point>& that )
      {
	if( this != &that )
	  {
	    weight_ = that.weight_;
	    //
	    vertices_[0] = that.vertices_[0];
	    vertices_[1] = that.vertices_[1];
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > Edge<Type_point>& 
      Edge<Type_point>::operator = ( Edge<Type_point>&& that )
      {
	if( this != &that )
	  {
	    // 
	    // 
	    weight_      = that.weight_;
	    vertices_[0] = std::move( that.vertices_[0] );
	    vertices_[1] = std::move( that.vertices_[1] );
	    // 
	    that.weight_      = static_cast<Type_point>(0.);	
	  }
	//
	//
	return *this;
      }
    /*!
     *  \brief Dump values for Edge
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     *  \param Edge : new vertices to add in the list
     */
    template< typename Type_point > 
      std::ostream& operator << ( std::ostream& stream, const Edge<Type_point>& that )
      {
	//
	//
	stream << "Vertices of the edge V1=" << that.V1() << "\n"
	       << "\" V2=" << that.V2() << "\n"
	       << "\" edge weight=\"" << that.weight() << std::endl;
	
	//
	//
	return stream;
      };
  }
}
#endif
