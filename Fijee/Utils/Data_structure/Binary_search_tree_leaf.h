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
#ifndef BINARY_SEARCH_TREE_LEAF_H
#define BINARY_SEARCH_TREE_LEAF_H
/*!
 * \file Binary_Search_Tree_Leaf.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <map>
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Binary_search_tree_leaf
   * \brief class representing a leaf for a binary search tree
   *
   *  This class representes a leaf for a binary search tree.
   */
  /*! \namespace Data_structure
   * 
   * Name space for a binary search tree. 
   * First leaf of the tree: level 0. 
   * For a depth of N levels, other leaf:  [0]  [1,   2, ..., N]
   *                                level 2^0, 2^1, 2^2 ... 2^N 
   * If num_of_level = 8, the last level (2^8) gathers 256 bins.
   */
  namespace Data_structure
  {
    class Binary_search_tree_leaf
    {
    private:
      //
      // Information carried by the leaf
      // 
      //! Information carried by the leaf
      int leaf_level_; 
      //! Cumulative intensity
      double cumulative_;
      //! Number of element on the leaf and bellow
      int cardinal_;
      //! Cluster number
      int cluster_number_;
      //! Left data for the tree
      Binary_search_tree_leaf* left_;
      //! Right data for the tree
      Binary_search_tree_leaf* right_;
      

    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Binary_search_tree_leaf
       *
       */
    Binary_search_tree_leaf():
      leaf_level_(0), cumulative_(0.), cardinal_(0), 
	cluster_number_(-1), left_(nullptr), right_(nullptr){};
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Binary_search_tree_leaf
       *
       */
    Binary_search_tree_leaf( const int Level ):
      leaf_level_(Level), cumulative_(0.), cardinal_(0), 
	cluster_number_(-1), left_(nullptr), right_(nullptr){};
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
      Binary_search_tree_leaf( const Binary_search_tree_leaf& ) = delete;
      /*!
       *  \brief Move Constructor
       *
       *  Constructor is a moving constructor
       *
       */
      Binary_search_tree_leaf( Binary_search_tree_leaf&& ) = delete;
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Binary_search_tree_leaf
       */
      virtual ~Binary_search_tree_leaf()
	{
	  // 
	  // leaves pointers
	  if(left_)
	    {
	      delete left_;
	      left_ = nullptr;
	    }
	  // 
	  if(right_)
	    {
	      delete right_;
	      right_ = nullptr;
	    }
	};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Binary_search_tree_leaf
       *
       */
      Binary_search_tree_leaf& operator = ( const Binary_search_tree_leaf& ) = delete;
      /*!
       *  \brief Move Operator =
       *
       *  Move operator of the class Binary_search_tree_leaf
       *
       */
      Binary_search_tree_leaf& operator = ( Binary_search_tree_leaf&& ) = delete;

    public:
      /*!
       *  \brief Accessor
       */
      //
      const int&  get_leaf_level_() const {return leaf_level_;};
      //
      double get_cumulative_() const {return cumulative_;};
      //
      int    get_cardinal_() const {return cardinal_;};
      //
      int    get_cluster_number_() const {return cluster_number_;};
      // 
      Binary_search_tree_leaf& get_left_() const {return *left_;};
      //
      Binary_search_tree_leaf& get_right_() const {return *right_;};

      // 
      // 
      void set_cluster_number_( int Num){cluster_number_ = Num;};
      // 
      void cardinal_iteration(){cardinal_++;};
      // 
      void cumulative_increase( double Adding_val){ cumulative_ += Adding_val;};
      
    public:
      /*!
       *  \brief Add left
       *
       *  This method add a leaf on the left.
       *
       */
      void add_left()
      {
	left_ = new Binary_search_tree_leaf(leaf_level_+1);
      };
      /*!
       *  \brief Add Right
       *
       *  This method add a leaf on the right.
       *
       */
      void add_right()
      {
	right_ = new Binary_search_tree_leaf(leaf_level_+1);
      };
      /*!
       *  \brief 
       *
       *  This method .
       *
       */
      void build_number_of_levels( const int Num_level )
      {
	if( leaf_level_ < Num_level /*+1: root leaf*/)
	  { 
	    // add a level
	    add_right();
	    add_left ();
	    // and keep going
	    left_-> build_number_of_levels(Num_level);
	    right_->build_number_of_levels(Num_level);	
	  }
      };
      /*!
       *  \brief 
       *
       *  This method .
       *
       */
      void print_level( const int Num_level )
      {
	if( leaf_level_ == Num_level )
	  {
	    std::cout 
	      << "leaf_level_: "  << leaf_level_
	      << " cumulative_: " << cumulative_ 
	      << " cardinal_: "   << cardinal_ 
	      << " cluster_number_: " << cluster_number_
	      << "\t Mu: " << (cardinal_ == 0 ? 0. : cumulative_ / (double)cardinal_)
	      << "\t %signal: " << ((double)cardinal_ / (256.*256.*256.)) * 100
	      << std::endl;
	  }
	else
	  {
	    // and keep going
	    left_-> print_level(Num_level);
	    right_->print_level(Num_level);	
	  }
      };
      /*!
       *  \brief Mark clusters
       *
       *  This method marks clusters when the reache 1% of the total signal.
       *
       */
      void clusterization( std::map<int, Binary_search_tree_leaf*>& Cardinal_map )
      {
	for( int level = 8 ; level > 0 ; level-- )
	  mark_clusters(level, Cardinal_map);
      };
      /*!
       *  \brief Mark clusters
       *
       *  This method marks clusters when the reache 1% of the total signal.
       *
       */
      void mark_clusters( const int Num_level, std::map<int, Binary_search_tree_leaf*>& Cardinal_map )
      {
	if( leaf_level_ == Num_level )
	  {
	    // 
	    double signal_percentage = ((double)cardinal_ / (256.*256.*256.)) * 100;
	    // 
	    if ( signal_percentage > 1. /* % */)
	      if ( !belongs_to_cluster() )
		{ 
		  cluster_number_ = -2; // temporary cluster number
		  Cardinal_map[cardinal_] = this;
		}
	  }
	else
	  {
	    // and keep going to the required level
	    left_-> mark_clusters(Num_level, Cardinal_map);
	    right_->mark_clusters(Num_level, Cardinal_map);	
	  }
      };
      /*!
       *  \brief Mark clusters
       *
       *  This method marks clusters when the reache 1% of the total signal.
       *
       */
      bool belongs_to_cluster()
      {
	if( leaf_level_ == 8 )
	  if( cluster_number_ != -1 ) return true;
	  else return false;
	else
	  if( cluster_number_ != -1 ) return true;
	  else
	    if( !left_-> belongs_to_cluster() )
	      right_->belongs_to_cluster();
	    else return true;
      };
      
    };
    /*!
     *  \brief Dump values for Binary_search_tree_leaf
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     *  \param Binary_search_tree_leaf : new position to add in the list
     */
    std::ostream& operator << ( std::ostream& stream, const Binary_search_tree_leaf& that )
      {
	stream 
	  << "leaf_level_: "  << that.get_leaf_level_()
	  << " cumulative_: " << that.get_cumulative_() 
	  << " cardinal_: "   << that.get_cardinal_() 
	  << " cluster_number_: " << that.get_cluster_number_()
	  << std::endl;

	// 
	// 
	return stream;
      };
  }
}
#endif
