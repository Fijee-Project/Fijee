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
#ifndef VERTEX_H
#define VERTEX_H
/*!
 * \file Vertex.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <memory>
//
//
//
#include <Utils/Data_structure/Basic_point.h>
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Data_structure
  {
    template< typename Type_point >
      class Edge;
    /*! \class Vertex
     * \brief classe representing a vertex
     *
     *  This class is a basic data structure for nodes in a ADT graph.
     */
    template< typename Type_point >
      class Vertex : public Basic_point< Type_point >
      {
	//! List of adjacent edges
	std::list< std::shared_ptr< Edge<Type_point> > > incident_edges_;
	//! Vertex index
	int index_;

      public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Vertex
	 *
	 */
	Vertex();
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Vertex
	 *
	 */
	Vertex( Type_point, Type_point, Type_point, int Index = -1, Type_point Weight = 1 );
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Vertex( const Vertex& );
	/*!
	 *  \brief Move Constructor
	 *
	 *  Constructor is a moving constructor
	 *
	 */
	Vertex( Vertex&& );
	/*!
	 *  \brief Destructeur
	 *
	 *  Destructor of the class Vertex
	 */
	virtual ~Vertex(){/*Do nothing*/};
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Vertex
	 *
	 */
	Vertex& operator = ( const Vertex& );
	/*!
	 *  \brief Move Operator =
	 *
	 *  Move operator of the class Vertex
	 *
	 */
	Vertex& operator = ( Vertex&& );
	/*!
	 *  \brief Operator ==
	 *
	 *  Operator compare of the class Vertex.
	 *
	 */
	bool operator == ( const Vertex& ) const;
	/*!
	 *  \brief Operator !=
	 *
	 *  Operator different of the class Vertex.
	 *
	 */
	bool operator != ( const Vertex& ) const;

      public:
	int get_index_()const{return index_;};

      public:
	/*!
	 *  \brief Incident edges
	 *
	 *  This method returns the list of edges incident on this vertex.
	 *
	 */
	std::list< std::shared_ptr< Edge<Type_point> > >& incident_edges(){return incident_edges_;};
	/*!
	 *  \brief is_adjacent_to
	 *
	 *  This method checks if this vertex is adjacent to V.
	 *
	 */
	bool is_adjacent_to( Vertex V ){};
	/*!
	 *  \brief Insert edge
	 *
	 *  This method inserts an edge to this vertex edges' list
	 *
	 */
	bool insert_edge( const Edge<Type_point>& E )
	{
	  bool edge_defined = false;
	  // 
	  for( auto edge : incident_edges_ )
	    if( edge->opposite_vertex(*this).squared_distance(E.opposite_vertex(*this)) == 0 )
	      edge_defined = true;

	  // 
	  // 
	  if( !edge_defined )
	    {
	      std::shared_ptr< Edge<Type_point> > incident_edge;
	      incident_edge.reset(&E);
	      incident_edges_.push_back( incident_edge );
	    }
	};
      };
    //
    //
    //
    template< typename Type_point >
      Vertex<Type_point>::Vertex(): Basic_point<Type_point>::Basic_point()
      {
      }
    //
    //
    //
    template< typename Type_point >
      Vertex<Type_point>::Vertex(Type_point X, Type_point Y, Type_point Z, int Index, Type_point Weight ): 
    Basic_point<Type_point>::Basic_point(X,Y,Z,Weight),
      index_(Index)
    {
    }
    //
    //
    //
    template< typename Type_point >
      Vertex<Type_point>::Vertex( const Vertex<Type_point>& that ): 
    Basic_point<Type_point>::Basic_point(that),
      index_(that.index_)
      {
      }
    //
    //
    //
    template< typename Type_point >
      Vertex<Type_point>::Vertex( Vertex<Type_point>&& that ): 
    Basic_point<Type_point>::Basic_point( std::move(that) ),
      index_(that.index_)
     {
       that.index_ = 0;
     }
    //
    //
    //
    template< typename Type_point > Vertex<Type_point>& 
      Vertex<Type_point>::operator = ( const Vertex<Type_point>& that )
      {
	// 
	// 
	Basic_point<Type_point>::operator = ( that );

	// 
	// 
	if( this != &that )
	  {
	    index_ = that.index_;
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > Vertex<Type_point>& 
      Vertex<Type_point>::operator = ( Vertex<Type_point>&& that )
      {
	// 
	// 
	Basic_point<Type_point>::operator = ( std::move(that) );

	if( this != &that )
	  {
	    index_      = that.index_;
	    that.index_ = 0;
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > bool
      Vertex<Type_point>::operator == ( const Vertex<Type_point>& that ) const
    {
      return ( (Basic_point<Type_point>::operator == (that)) );
    }
    //
    //
    //
    template< typename Type_point > bool
      Vertex<Type_point>::operator != ( const Vertex<Type_point>& that ) const
    {
      return ( (Basic_point<Type_point>::operator != (that)) );
    }
    /*!
     *  \brief Dump values for Vertex
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     *  \param Vertex : new position to add in the list
     */
    template< typename Type_point > 
      std::ostream& operator << ( std::ostream& stream, const Vertex<Type_point>& that )
      {
	//
	//
	stream 
	  << "x= " << that.x() 
	  << " y= " << that.y() 
	  << " z= " << that.z()
	  << " index= " << that.get_index_()
	  << " weight= " << that.weight();
	
	//
	//
	return stream;
      };
  }
}
#endif
