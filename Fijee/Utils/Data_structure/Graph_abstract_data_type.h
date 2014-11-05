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
#ifndef GRAPH_ABSTRACT_DATA_TYPE_H
#define GRAPH_ABSTRACT_DATA_TYPE_H
/*!
 * \file Graph_abstract_data_type.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <string>
#include <list>
#include <memory>
#include <algorithm>
//
// UCSF
//
#include "Fijee/Fijee_exception_handler.h"
#include "Utils/Data_structure/Vertex.h"
#include "Utils/Data_structure/Edge.h"
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  namespace Data_structure
  {
    /*! \class Graph_abstract_data_type
     * \brief classe representing a graph abstract data type (ADT)
     *
     *  This class is graph abstract data type (ADT), which is suitable for undirected graphs, that is, graphs whose edges are all undirected.
     * In this context Fijee implements a Adjacency Matrix Structure
     *
     */
    template< typename Type_point >
      class Graph_abstract_data_type
      {
      private:
	//! Vertices list
	std::list< Vertex<Type_point> > vertices_;
	//! Edge list
	std::list< Edge<Type_point> > edges_;
	//! Matrix strcture holding the edges reference.
	std::shared_ptr< Edge<Type_point> >** adjacency_matrix_;


      public:
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Graph_abstract_data_type
	 *
	 */
	Graph_abstract_data_type();
	/*!
	 *  \brief Default Constructor
	 *
	 *  Constructor of the class Graph_abstract_data_type
	 *
	 */
//	Graph_abstract_data_type( const Vertex<Type_point> &, const Vertex<Type_point> &, 
//	      Type_point Weight = static_cast<Type_point>(0.) );
	/*!
	 *  \brief Copy Constructor
	 *
	 *  Constructor is a copy constructor
	 *
	 */
	Graph_abstract_data_type( const Graph_abstract_data_type& );
	/*!
	 *  \brief Move Constructor
	 *
	 *  Constructor is a moving constructor
	 *
	 */
	Graph_abstract_data_type( Graph_abstract_data_type&& );
	/*!
	 *  \brief Destructeur
	 *
	 *  Destructor of the class Graph_abstract_data_type
	 */
	virtual ~Graph_abstract_data_type(){/*Do nothing*/};
	/*!
	 *  \brief Operator =
	 *
	 *  Operator = of the class Graph_abstract_data_type
	 *
	 */
	Graph_abstract_data_type& operator = ( const Graph_abstract_data_type& );
	/*!
	 *  \brief Move Operator =
	 *
	 *  Move operator of the class Graph_abstract_data_type
	 *
	 */
	Graph_abstract_data_type& operator = ( Graph_abstract_data_type&& );
	/*!
	 *  \brief Move Operator ()
	 *
	 *  operator () of the class Graph_abstract_data_type
	 *
	 */
	bool operator () ( std::shared_ptr< Edge<Type_point> >& );

      public:
	const std::list< Vertex<Type_point> >& get_vertices_()         const {return vertices_; }
	const std::list< Edge<Type_point> >&   get_edges_()            const {return edges_; }
	std::shared_ptr< Edge<Type_point> >**  get_adjacency_matrix_() const {return adjacency_matrix_; }

      public:
	/*!
	 *  \brief Insert vertex(x)
	 *
	 *  This method inserts a new vertex storing element x.
	 *
	 */
	bool insert_vertex( const Vertex<Type_point>& V )
	{
	  vertices_.push_back(V);
	};
	/*!
	 *  \brief Insert Edge
	 *
	 *  This method inserts and return a new undirected edge with end vertices v and w and storing element x.
	 *
	 */
	bool insert_edge( const Vertex<Type_point>&, const Vertex<Type_point>& );
	/*!
	 *  \brief Erase vertex
	 *
	 *  This method removes vertex V and all its incident edges.
	 *
	 */
	bool erase_vertex(const Vertex<Type_point>&){};
	/*!
	 *  \brief Erase edge
	 *
	 *  This method Remove edge E.
	 *
	 */
	bool eraseEdge( const Edge<Type_point>& ){};
 	/*!
	 *  \brief Initialize the adjacency matrix
	 *
	 *  This method initializes the adjacency matrix
	 *
	 */
	bool init_adjacency_matrix()
	{
	  try
	    {
	      if( !vertices_.empty() )
		{
		  adjacency_matrix_ = new std::shared_ptr< Edge<Type_point> >*[vertices_.size()];
		  // 
		  for( int i = 0 ; i < vertices_.size() ; i++ )
		    adjacency_matrix_[i] = new std::shared_ptr< Edge<Type_point> >[vertices_.size()];
		}
	      else
		{
		  std::string message = std::string("The vertices list must be initialized first!");
		  //
		  throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
		}
	    }
	  catch( Fijee::Exception_handler& err )
	    {
	      std::cerr << err.what() << std::endl;
	    }
	};
      };
    //
    //
    //
    template< typename Type_point >
      Graph_abstract_data_type<Type_point>::Graph_abstract_data_type()
      {
      }
    //
    //
    //
    template< typename Type_point >
      Graph_abstract_data_type<Type_point>::Graph_abstract_data_type( const Graph_abstract_data_type<Type_point>& that )
      {
	vertices_ = that.vertices_;
	edges_    = that.edges_;
	init_adjacency_matrix();
      }
    //
    //
    //
    template< typename Type_point >
      Graph_abstract_data_type<Type_point>::Graph_abstract_data_type( Graph_abstract_data_type<Type_point>&& that ):
    adjacency_matrix_(nullptr)
      {
	vertices_         = std::move( that.vertices_ );
	edges_            = std::move( that.edges_ );
	adjacency_matrix_ = that.adjacency_matrix_;
	// 
	that.adjacency_matrix_ = nullptr;
      }
    //
    //
    //
    template< typename Type_point > Graph_abstract_data_type<Type_point>& 
      Graph_abstract_data_type<Type_point>::operator = ( const Graph_abstract_data_type<Type_point>& that )
      {
	if( this != &that )
	  {
	    vertices_ = that.vertices_;
	    edges_    = that.edges_;
	    init_adjacency_matrix();
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > Graph_abstract_data_type<Type_point>& 
      Graph_abstract_data_type<Type_point>::operator = ( Graph_abstract_data_type<Type_point>&& that )
      {
	if( this != &that )
	  {
	    adjacency_matrix_ = nullptr;
	    vertices_         = std::move( that.vertices_ );
	    edges_            = std::move( that.edges_ );
	    adjacency_matrix_ = that.adjacency_matrix_;
	    // 
	    that.adjacency_matrix_ = nullptr;
	  }
	//
	//
	return *this;
      }
    //
    //
    //
    template< typename Type_point > bool
      Graph_abstract_data_type<Type_point>::operator () ( std::shared_ptr< Edge<Type_point> >& that )
      {
	
      }
    //
    //
    //
    template< typename Type_point > bool
      Graph_abstract_data_type<Type_point>::insert_edge( const Vertex<Type_point>& V1, 
							 const Vertex<Type_point>& V2 )
      {
	try
	  {
	    // 
	    // Points belong to the graph
	    auto v1 = find(vertices_.begin(), vertices_.end(), V1);
	    auto v2 = find(vertices_.begin(), vertices_.end(), V2);
	    // 
	    if( v1 == vertices_.end() || v2  == vertices_.end() )
	      {
		std::string message = std::string("One or both vertices do not belong the graph!");
		  //
		throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	      }

	    // 
	    // Check if the edge does not already exist
	    bool edge_defined = false;
	    // 
	    for( auto edge : edges_ )
	      if( edge.is_incident_on(V1) && edge.is_incident_on(V2) )
		edge_defined = true;
	    
	    // 
	    if( !edge_defined )
	      {
		edges_.push_back( Edge<Type_point>(*v1, *v2) );
		// up date of the adjacency_matrix_
		if( adjacency_matrix_ )
		  {
//		    auto last_edge = edges_.back();
		    adjacency_matrix_[V1.get_index_()][V2.get_index_()].reset( new Edge<Type_point>(*v1, *v2) );
		    adjacency_matrix_[V2.get_index_()][V1.get_index_()] = adjacency_matrix_[V1.get_index_()][V2.get_index_()];
		    // add the edge in the vertices' edges list
//		    v1->insert_edge(last_edge);
//		    v2->insert_edge(last_edge);


//		    std::cout << "Make Edge" << std::endl;
//		    std::cout << V1 << " " << V2 << std::endl;
//		    std::cout << V1.get_index_() << " " << V2.get_index_() << std::endl;
//		    std::cout << (adjacency_matrix_[V1.get_index_()][V2.get_index_()]->get_vertices_())[0].get_index_()
//			      << " "
//			      << (adjacency_matrix_[V1.get_index_()][V2.get_index_()]->get_vertices_())[1].get_index_()
//			      << std::endl;


		  }
		else
		  {
		    std::string message = std::string("The adjacency_matrix_ must be initialized first!");
		    //
		    throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
		  }
	      }
	    else
	      {
		std::string message = std::string("This edge was already created");
		//
		//		throw Fijee::Warning_handler( message,  __LINE__, __FILE__ );
	      }
	  }
	catch( Fijee::Exception_handler& err )
	  {
	    std::cerr << err.what() << std::endl;
	  }
      }
    /*!
     *  \brief Dump values for Graph_abstract_data_type
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     *  \param Graph_abstract_data_type : new vertices to add in the list
     */
    template< typename Type_point > 
      std::ostream& operator << ( std::ostream& stream, const Graph_abstract_data_type<Type_point>& that )
      {
	//
	//
	for ( int i = 0 ; i < that.get_vertices_().size() ; i++ )
	  for ( int j = i+1 ; j < that.get_vertices_().size() ; j++ )
	    if( (that.get_adjacency_matrix_())[i][j] )
	      stream << *(that.get_adjacency_matrix_())[i][j] << std::endl;
	
	//
	//
	return stream;
      };
  }
}
#endif
