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
#ifndef GRAY_LEVEL_BINARY_SEARCH_TREE_H
#define GRAY_LEVEL_BINARY_SEARCH_TREE_H
/*!
 * \file Gray_level_binary_search_tree.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <map>
#include <tuple>
#include <bitset> 
// 
// UCSF
// 
#include "Utils/Data_structure/Binary_search_tree_leaf.h"
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Gray_level_binary_search_tree
   * \brief class representing gray level binary search tree
   *
   *  This class clusteries the gray level voxel
   */
  /*! \namespace Data_structure
   * 
   * Name space for our new package
   *
   */
  namespace Data_structure
  {
    template< typename Type, int num_of_level = 1 >
      class Gray_level_binary_search_tree
      {
      private:
      //! Binary tree data structure      
      Binary_search_tree_leaf tree_;
      std::map< int, Binary_search_tree_leaf* > clustering_;

      public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Gray_level_binary_search_tree
       *
       */
      Gray_level_binary_search_tree();
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
      Gray_level_binary_search_tree( const Gray_level_binary_search_tree& ) = delete;
      /*!
       *  \brief Move Constructor
       *
       *  Constructor is a moving constructor
       *
       */
      Gray_level_binary_search_tree( Gray_level_binary_search_tree&& ) = delete;
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Gray_level_binary_search_tree
       */
      virtual ~Gray_level_binary_search_tree(){ /*Do nothing*/ };
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Gray_level_binary_search_tree
       *
       */
      Gray_level_binary_search_tree& operator = ( const Gray_level_binary_search_tree& ) = delete;
      /*!
       *  \brief Move Operator =
       *
       *  Move operator of the class Gray_level_binary_search_tree
       *
       */
      Gray_level_binary_search_tree& operator = ( Gray_level_binary_search_tree&& ) = delete;

      public:
      /*!
       *  \brief
       *
       *  
       *
       */
      void push( const Type& );
      /*!
       *  \brief
       *
       *  
       *
       */
      int cluster( const Type& Voxel);
      /*!
       *  \brief 
       *
       *  
       *
       */
      void print_level( const int Level ){ tree_.print_level(Level);};
      /*!
       *  \brief 
       *
       *  
       *
       */
      void mark_clusters( const int );
      };
    // 
    // 
    // 
    template <  typename Type, int num_of_level >
      Gray_level_binary_search_tree<Type, num_of_level>::Gray_level_binary_search_tree()
      {
	// Build the tree over num_of_level levels
	tree_.build_number_of_levels(num_of_level);
      }
    // 
    // 
    // 
    template <  typename Type, int num_of_level > int
      Gray_level_binary_search_tree<Type, num_of_level>::cluster( const Type& Voxel)
      {
	// 
	// Convert intensity in binary
	std::bitset< num_of_level > voxel_intensity ( static_cast<int>(Voxel) );
	// 
	Binary_search_tree_leaf* tempo_leaf = &tree_;
	// 
	// 128: 10000000
	// [0] = 0 
	// [1] = 0 
	// ...
	// [7] = 1 
	// 
	int level = 0;
	while( level < num_of_level )
	  {
	    if ( tempo_leaf->get_cluster_number_() != -1 )
	      return tempo_leaf->get_cluster_number_();
	    else if( &tempo_leaf->get_right_() != nullptr )
	      {
		if( voxel_intensity[num_of_level - 1 - level] == 1 )
		  tempo_leaf = &(tempo_leaf->get_right_());
		else
		  tempo_leaf = &(tempo_leaf->get_left_());
	      }
	    else
	      std::cerr << "We have a problem" << std::endl;
	    //
	    level++;
	  }
	// Last level
	if ( tempo_leaf->get_cluster_number_() != -1 )
	  return tempo_leaf->get_cluster_number_();
	else
	  return -1;
      }
    // 
    // 
    // 
    template <  typename Type, int num_of_level > void
      Gray_level_binary_search_tree<Type, num_of_level>::push( const Type& Voxel)
      {
	// 
	// Convert intensity in binary
	std::bitset< num_of_level > voxel_intensity ( static_cast<int>(Voxel) );
	// 
	Binary_search_tree_leaf* tempo_leaf = &tree_;
	// 
	// 128: 10000000
	// [0] = 0 
	// [1] = 0 
	// ...
	// [7] = 1 
	// 
	int level = 0;
	while( level < num_of_level )
	  {
	  tempo_leaf->cumulative_increase(static_cast<double>(Voxel));
	  tempo_leaf->cardinal_iteration();
	  //
	  if( voxel_intensity[num_of_level - 1 - level] == 1 )
	    tempo_leaf = &(tempo_leaf->get_right_());
	  else
	    tempo_leaf = &(tempo_leaf->get_left_());
	  //
	  level++;
	}
	// Last level
	tempo_leaf->cumulative_increase(static_cast<double>(Voxel));
	tempo_leaf->cardinal_iteration();

	// for the control
	tempo_leaf = &(tempo_leaf->get_right_());
	
	// 
	// 
	if( tempo_leaf != nullptr )
	  std::cerr << "We have a problem" << std::endl;
      }
    // 
    // 
    // 
    template <  typename Type, int num_of_level > void
      Gray_level_binary_search_tree<Type, num_of_level>::mark_clusters( const int Level )
    { 
      // 
      // Enumeration of potential clusters
      tree_.mark_clusters(Level, clustering_);
      
      // 
      //
      int num_of_clusters = clustering_.size();
      int cluster = 0;
      // 
      for ( auto cl = clustering_.rbegin() ; cl != clustering_.rend() ; cl++ )
	cl->second->set_cluster_number_(cluster++);
    };

    //    /*!
    //     *  \brief Dump values for Gray_level_binary_search_tree
    //     *
    //     *  This method overload "<<" operator for a customuzed output.
    //     *
    //     *  \param Gray_level_binary_search_tree : new position to add in the list
    //     */
    //    std::ostream& operator << ( std::ostream&, const Gray_level_binary_search_tree& );
  }
}
#endif
