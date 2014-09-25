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
#include "Parcellation_Scotch.h"
// 
// 
// 
extern "C"
{
#include <stdlib.h>     /* calloc, exit, free */
  //#include <scotch.h>
}
//
// We give a comprehensive type name
//
typedef Domains::Parcellation_Scotch DPS;
typedef Domains::Access_parameters DAp;
//
//
//
DPS::Parcellation_Scotch( const C3t3& Mesh, const Cell_pmap& Mesh_map, 
			  const Brain_segmentation Segment, const int N_partitions ):
  Parcellation(),
  segment_( Segment ),  n_partitions_( N_partitions )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Parcellation_Scotch::Parcellation_Scotch");

  //
  // 
  int 
    vertex_new_id = 0,
    cell_id = 0;
  // 
  for( Cell_iterator cit = Mesh.cells_in_complex_begin() ;
       cit != Mesh.cells_in_complex_end() ; cit++ )
    if( Mesh_map.subdomain_index( cit ) == Segment )
      {
	// 
	// record the address of the cell in the Mesh
	elements_nodes_.push_back( cit );
	int mesh_cell = cell_id++;
	
	// 
	// 
	for( int i = 0 ; i < 4 ; i++ )
	  {
	    // 
	    // 
	    auto it_v_handler = edge_vertex_to_element_.find(cit->vertex(i));
	    // The vertex is not in the edge_vertex_to_element map
	    if ( it_v_handler == edge_vertex_to_element_.end() )
	      {
		edge_vertex_to_element_.
		  insert( std::pair<Tr::Vertex_handle,
			  std::tuple<int, std::list<int> > >
			  (cit->vertex(i),
			   std::make_tuple (vertex_new_id++,
					    std::list<int>(1, mesh_cell))) );
	      }
	    // The vertex is in the edge_vertex_to_element map
	    else
	      {
		(std::get<1/*list of cell id*/>(it_v_handler->second)).push_back(mesh_cell);
	      }
	  }
      }

  // 
  // 
  elements_partitioning_.resize( elements_nodes_.size() );
  nodes_partitioning_.resize( edge_vertex_to_element_.size() );
}
//
//
//
DPS::Parcellation_Scotch( const DPS& that ):
  Parcellation(),
  segment_( that.segment_ ), n_partitions_( that.n_partitions_ ),
  elements_nodes_(that.elements_nodes_), edge_vertex_to_element_(that.edge_vertex_to_element_)
{
}
//
//
//
DPS::~Parcellation_Scotch()
{
  /* Do nothing */
}
//
//
//
void 
DPS::Mesh_partitioning(  )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Parcellation_Scotch::Mesh_partitioning");

//  // 
//  // Build Scotch graph (Scotch and libScotch 6.0 User's Guide)
//  //
//
//  // Initialization
//  // Base value for element indexings.
//  SCOTCH_Num velmbas = 0;
//  // Base value for node indexings. 
//  // The base value of the underlying graph, baseval, is set as min(velmbas, vnodbas).
//  SCOTCH_Num vnodbas = static_cast< SCOTCH_Num >( elements_nodes.size() );
//  // Number of element vertices in mesh.
//  SCOTCH_Num velmnbr = vnodbas; //static_cast< SCOTCH_Num >( elements_nodes.size() );
//  // Number of node vertices in mesh. 
//  // The overall number of vertices in the underlying graph, vertnbr, is set as velmnbr + vnodnbr.
//  SCOTCH_Num vnodnbr = static_cast< SCOTCH_Num >( edge_vertex_to_element.size() );
//  // Number of arcs in mesh. 
//  // Since edges are represented by both of their ends, the number of edge data in the mesh is 
//  // twice the number of edges.
//  SCOTCH_Num edgenbr = 2 * 4 /*vertices in cell*/ * velmnbr;
//  // Array of start indices in edgetab of vertex (that is, both elements and nodes) 
//  // adjacency sub-arrays.
//  // verttab[baseval /*0*/+ vertnbr] = (baseval + edgenbr) = edgenbr
//  SCOTCH_Num* verttab;
//  verttab = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), velmnbr + vnodnbr);
//  // Array of after-last indices in edgetab of vertex adjacency sub-arrays. 
//  // For any element or node vertex i, with baseval i < (baseval + vertnbr), 
//  // vendtab[i] − verttab[i] is the degree of vertex i, and the indices of the neighbors of i 
//  // are stored in edgetab from edgetab[verttab[i]] to edgetab [vendtab[i]−1], inclusive.
//  SCOTCH_Num* vendtab;
//  vendtab = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), velmnbr + vnodnbr);
//  // SCOTCH_Num* vendtab = NULL;
//  //  SCOTCH_Num vendtab[velmnbr + vnodnbr];
//  //  vendtab = (verttab + 1);
//
//  // 
//  // Bipartite element-node graph construction
//  // 
//
//
//  std::cout << "0 step: inum = " << inum << "then become 0" << std::endl;
//  std::cout << "vnodbas = " << vnodbas <<std::endl;
//  std::cout << "velmnbr = " << velmnbr <<std::endl;
//  std::cout << "vnodnbr = " << vnodnbr <<std::endl;
//  std::cout << "edgenbr = " << edgenbr <<std::endl;
//  
//  // 
//  // 
//  std::list< SCOTCH_Num > edgetab_list;
//  
//  // 
//  // First, we fill the elements
//  inum = 0; 
//  //
//  for ( Cell_iterator cit : elements_nodes )
//    {
//      // 
//      // 
//      int cell_id = inum++;
//      // Each element (cell) has 4 vertices
//      verttab[ cell_id ] = 4 * cell_id;
//      vendtab[ cell_id ] = 4 * cell_id + 4;
//      std::cout << "verttab[" <<  cell_id << " ] = " << verttab[ cell_id ] << std::endl;
//      std::cout << "vendtab[" <<  cell_id << " ] = " << vendtab[ cell_id ] << std::endl;
//      // 
//      // 
//      for ( int i = 0 ; i < 4 ; i++ )
//	{
//	  // 
//	  auto it_vertex = edge_vertex_to_element.find( cit->vertex(i) );
//	  // 
//	  if( it_vertex != edge_vertex_to_element.end() )
//	    {
//	      // 
//	      // we shift the vertex id after the element number (velmnbr)
//	      // edgetab[4*cell_id+i]=velmnbr+static_cast<SCOTCH_Num>(std::get<0>(it_vertex->second));
//	      edgetab_list.push_back(velmnbr+static_cast<SCOTCH_Num>(std::get<0>(it_vertex->second)));
//	    }
//	  else
//	    {
//	      std::cerr << "Parcelletion: all vertices must be found:" << std::endl;
//	      std::cerr << it_vertex->first->point() << std::endl;
//	      abort();
//	    }
//	}
//    }
//
//      std::cout << "First step: inum = " << inum <<std::endl;
//      std::cout << "vnodbas = " << vnodbas <<std::endl;
//      std::cout << "velmnbr = " << velmnbr <<std::endl;
//      std::cout << "vnodnbr = " << vnodnbr <<std::endl;
//      std::cout << "edgenbr = " << edgenbr <<std::endl;
//      std::cout << "edgenbr_list = 4*velmnbr = " << edgetab_list.size() <<std::endl;
//
//  
//  // 
//  // Second, we fill the nodes
//  int edgetab_vertex_pos = 4*inum;
//  std::cout << "vertex start pos = " <<  edgetab_vertex_pos<<std::endl;
//  for ( auto it_vertex : edge_vertex_to_element )
//    {
//      // 
//      // 
//      verttab[ inum ]     = edgetab_vertex_pos;
//      std::cout << "verttab[" <<  inum  << " ] = " << verttab[ inum ] << std::endl;
//      edgetab_vertex_pos += static_cast<SCOTCH_Num>( std::get<1>(it_vertex.second).size() );
//      vendtab[ inum++ ]   = edgetab_vertex_pos;
//      std::cout << "vendtab[" <<  inum - 1 << " ] = " << vendtab[ inum - 1 ] 
//		<< std::endl;
//      //
//      for ( auto cell_id : std::get<1>(it_vertex.second) )
//	{
//	  // edgetab_vertex_pos += element_connected++; 
//	  // edgetab[edgetab_vertex_pos] = static_cast<SCOTCH_Num>(cell_id);
//	  edgetab_list.push_back(static_cast<SCOTCH_Num>(cell_id));
//	}
//    }
//  // last element
//  //  verttab[ inum ] = edgetab_vertex_pos;
//
//      std::cout << "Second step: inum = " << inum <<std::endl;
//      std::cout << "vnodbas = " << vnodbas <<std::endl;
//      std::cout << "velmnbr = " << velmnbr <<std::endl;
//      std::cout << "vnodnbr = " << vnodnbr <<std::endl;
//      std::cout << "edgenbr = " << edgenbr <<std::endl;
//      std::cout << "edgenbr_list = " << edgetab_list.size() <<std::endl;
//
//      std::cout << "copy edgetab_list in edgetab" <<std::endl;
//  
//  // 
//  // edgetab is the adjacency array, of size at least edgenbr 
//  // (it can be more if the edge array is not compact).
//  SCOTCH_Num* edgetab;
//  edgetab = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), edgetab_list.size());
//  std::copy(edgetab_list.begin(), edgetab_list.end(), edgetab);
//
//  // 
//  if( vendtab[ --inum ] != edgenbr )
//    {
//      std::cerr << "vnodbas = " << vnodbas <<std::endl;
//      std::cerr << "velmnbr = " << velmnbr <<std::endl;
//      std::cerr << "vnodnbr = " << vnodnbr <<std::endl;
//      std::cerr << "edgenbr = " << edgenbr <<std::endl;
//      std::cerr << "verttab[vertnbr] = " << verttab[velmnbr+vnodnbr] <<std::endl;
//      std::cerr << "Parcelletion: we should have: verttab[vertnbr] = edgenbr" << std::endl;
//      abort();
//    }
//
//  // 
//  // Scotch Mesh handling
//  SCOTCH_Num* velotab = NULL;
//  SCOTCH_Num* vlbltab = NULL;
//  SCOTCH_Num* vnlotab = NULL;
//  // Graph construction
//  SCOTCH_Mesh* meshptr;
//  meshptr = SCOTCH_meshAlloc();
//  //  meshptr = ((SCOTCH_Mesh *) memAlloc (sizeof (SCOTCH_Mesh)));
//  std::cout << "Init the Scotch mesh" <<std::endl;
//  if ( SCOTCH_meshInit( meshptr ) != 0 )
//    {
//      std::cerr << "Parcelletion: SCOTCH_meshInit abort" << std::endl;
//      abort();
//    }
//  //
//  std::cout << "Build the Scotch mesh" <<std::endl;
//  if ( SCOTCH_meshBuild( meshptr, velmbas, vnodbas, velmnbr, vnodnbr,
//			 verttab, vendtab, velotab, vnlotab, vlbltab, 
//			 edgenbr, edgetab) != 0 )
//    {
//      std::cerr << "Parcelletion: SCOTCH_meshBuild abort" << std::endl;
//      abort();
//    }
//
//
//#ifdef FIJEE_TRACE
//  // 
//  // At least in the development phase, it is recommended to check the Scotch mesh
//  // SCOTCH_meshCheck
//  std::cout << "Check the Scotch mesh" <<std::endl;
////  if ( SCOTCH_meshCheck( meshptr ) != 0 )
////    {
////      std::cerr << "Parcelletion: SCOTCH_meshCheck abort" << std::endl;
////      abort();
////    }
//#endif      
//  
//  // 
//  // Graph construction
//  SCOTCH_Graph* grafptr;
//  grafptr = SCOTCH_graphAlloc();
//  SCOTCH_graphInit( grafptr );
//  // 
//  std::cout << "Build graph from the Scotch mesh" <<std::endl;
//  if ( SCOTCH_meshGraph( meshptr, grafptr ) != 0 )
//    {
//      std::cerr << "Parcelletion: SCOTCH_meshGraph abort" << std::endl;
//      abort();
//    }
//
//  //
//  // Partitioning strategy
//  SCOTCH_Strat* strat;
//  strat = SCOTCH_stratAlloc();
//  SCOTCH_stratInit(strat);
//  //
//  SCOTCH_Num* vertices_partition;
//  vertices_partition = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), vnodnbr);
// 
//  // Partition graph
//  std::cout << "Graph partitioning" <<std::endl;
//  if (SCOTCH_graphPart(grafptr, 16, strat, vertices_partition))
//  {
//    std::cerr << "Parcelletion: SCOTCH_graphPart abort" << std::endl;
//    abort();
//  }
//  std::cout << "VERTEX LIST" <<std::endl;
//  std::cout << "X Y Z PAR" <<std::endl;
//  for ( auto vertex : edge_vertex_to_element )
//    std::cout << (vertex.first)->point() << " " 
//	      << vertices_partition[std::get<0>(vertex.second)]
//	      << std::endl;
//
//
//  std::cout << "Free Scotch objects" <<std::endl;
//  // 
//  // Free the structures
//  // Array and SCOTCH_meshExit
//  SCOTCH_meshExit(meshptr);
//  //
//  SCOTCH_graphExit(grafptr);
//  // 
//  SCOTCH_stratExit(strat);
//  //
//  free(verttab);
//  free(edgetab);

  //
  //
  Make_analysis();
}
//
//
//
void 
DPS::Make_analysis()
{
#ifdef FIJEE_TRACE
#if FIJEE_TRACE == 100
  //
  // Warning: output are raw output from the data framework. To be in the good framework 
  // its needs the usual transformation from Access_parameters
  // 
  output_stream_
    // Centroid with region
    << "X Y Z Region " 
    << std::endl;

  // 
  // 
  for( int cell = 0 ; cell < static_cast<int>(elements_nodes_.size()) ; cell++ )
    {
      // load the cell
      auto centroid = elements_nodes_[cell];
      // 
      Point_3 
	CGAL_cell_vertices[5];
      // 
      for (int i = 0 ; i < 4 ; i++)
	  CGAL_cell_vertices[i] = centroid->vertex( i )->point();

      // 
      // Compute centroid
      CGAL_cell_vertices[4] = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);

      // 
      // Stream the result
      output_stream_ << CGAL_cell_vertices[4].point() << " " 
		     << elements_partitioning_[cell] 
		     << std::endl;
    }
   

  //
  //
  std::string file_name("Parcellation_Scotch_segment_");
  file_name += std::to_string(static_cast<int>(segment_));
  file_name += ".frame";
  // 
  Make_output_file( file_name.c_str() );
#endif
#endif      
}
//
//
//
DPS& 
DPS::operator = ( const DPS& that )
{

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DPS& that)
{

  //
  //
  return stream;
};
