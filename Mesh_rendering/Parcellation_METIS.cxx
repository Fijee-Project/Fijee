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
#include "Parcellation_METIS.h"
// 
// 
// 
extern "C"
{
#include <stdlib.h>     /* calloc, exit, free */
#include <metis.h>
}
//
// We give a comprehensive type name
//
typedef Domains::Parcellation_METIS DPM;
typedef Domains::Access_parameters DAp;
//
//
//
DPM::Parcellation_METIS( const C3t3& Mesh, const Cell_pmap& Mesh_map, 
			 const Brain_segmentation Segment, const int N_partitions ):
  Parcellation(),
  segment_( Segment ),  n_partitions_( N_partitions )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Parcellation_METIS::Parcellation_METIS");

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
DPM::Parcellation_METIS( const DPM& that ):
  Parcellation(),
  segment_( that.segment_ ), n_partitions_( that.n_partitions_ ),
  elements_nodes_(that.elements_nodes_), edge_vertex_to_element_(that.edge_vertex_to_element_)
{
}
//
//
//
DPM::~Parcellation_METIS()
{
  /* Do nothing */
}
//
//
//
void 
DPM::Mesh_partitioning()
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Parcellation_METIS::Mesh_partitioning");
  //
  std::cout << "Build Metis mesh data structure" << std::endl;

  // 
  // Metis data structure
  // 

#ifdef TRACE
  printf("If not compliance, you are adviced to change IDXTYPEWIDTH metis.h\n.");
  printf("Size of idx_t: %zubits, real_t: %zubits, idx_t *: %zubits\n", 
	 8*sizeof(idx_t), 8*sizeof(real_t), 8*sizeof(idx_t *));
#endif
  
  // 
  // 
  idx_t options[METIS_NOPTIONS];
  // 
  METIS_SetDefaultOptions(options);
//  // METIS_PTYPE_RB || METIS_PTYPE_KWAY
//  options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
//  // Specifies the type of objective. Possible values are:
//  // - METIS_OBJTYPE_CUT Edge-cut minimization.
//  // - METIS_OBJTYPE_VOL Total communication volume minimization.
//  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
//  // Specifies the matching scheme to be used during coarsening. Possible values are:
//  // - METIS_CTYPE_RM Random matching.
//  // - METIS_CTYPE_SHEM Sorted heavy-edge matching.
//  options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
////  // Determines the algorithm used during initial partitioning. Possible values are:
////  // - METIS_IPTYPE_GROW Grows a bisection using a greedy strategy.
////  // - METIS_IPTYPE_RANDOM Computes a bisection at random followed by a refinement.
////  // - METIS_IPTYPE_EDGE Derives a separator from an edge cut.
////  // - METIS_IPTYPE_NODE Grow a bisection using a greedy node-based strategy.
////  options[METIS_OPTION_IPTYPE]  = METIS_IPTYPE_GROW;
//  // Determines the algorithm used for refinement. Possible values are:
//  // - METIS_RTYPE_FM FM-based cut refinement.
//  // - METIS_RTYPE_GREEDY Greedy-based cut and volume refinement.
//  // - METIS_RTYPE_SEP2SIDED Two-sided node FM refinement.
//  // - METIS_RTYPE_SEP1SIDED One-sided node FM refinement.
//  options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
//  // Debug
//  options[METIS_OPTION_DBGLVL] = 0;
//  //
//  //  options[METIS_OPTION_UFACTOR] = params->ufactor;
//  // 0 || 1
//  options[METIS_OPTION_MINCONN] = 0;
//  // 0 || 1
//  options[METIS_OPTION_CONTIG] = 0;
//  // Specifies the seed for the random number generator.
//  options[METIS_OPTION_SEED] = -1;
//  // Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process. Default is 10.
//  options[METIS_OPTION_NITER] = 10;
//  // Specifies the number of different partitionings that it will compute. The final partitioning 
//  // is the one that achieves the best edgecut or communication volume. Default is 1.
//  options[METIS_OPTION_NCUTS] = 1;
//  //
//  //  options[METIS_OPTION_NUMBERING] = 0;

  // 
  // number of elements and vertices
  idx_t 
    elem_nbr     = static_cast<idx_t>(elements_nodes_.size()),
    vertices_nbr = static_cast<idx_t>(edge_vertex_to_element_.size());
  // The size of the eptr array is n + 1, where n is the number of elements in the mesh. 
  // The size of the eind array is of size equal to the sum of the number of nodes in all 
  // the elements of the mesh. The list of nodes belonging to the ith element of the mesh 
  // are stored in consecutive locations of eind starting at position eptr[i] up to 
  // (but not including) position eptr[i+1].
  idx_t *eptr, *epart, *eind, *npart;
  //
  eptr  = (idx_t*)calloc(sizeof(idx_t), elem_nbr+1);
  eind  = (idx_t*)calloc(sizeof(idx_t), 4 * elem_nbr);
  //partition vector for the elements of the mesh
  epart = (idx_t*)calloc(sizeof(idx_t), elem_nbr);
  // partition vector for the nodes of the mesh
  npart = (idx_t*)calloc(sizeof(idx_t), vertices_nbr);

  // 
  // Build Metis mesh data structure
  // 
     
  // 
  // Load elements in eptr (element ptr) and eind (element index) arrays
  for ( int elem = 0 ; elem < elem_nbr ; elem++ )
    {
      eptr[elem] = 4*elem;
      // 
      for ( int num = 0 ; num < 4 ; num++ )
	{
	  // 
	  auto vertex = edge_vertex_to_element_.find(elements_nodes_[elem]->vertex(num));
	  //
	  if( vertex != edge_vertex_to_element_.end() )
	    eind[4*elem+num] = std::get<0>(vertex->second);
	  else
	    {
	      std::cerr << "Parcelletion: all vertices must be found:" << std::endl;
	      std::cerr << vertex->first->point() << std::endl;
	      abort();
	    }
	}
    }
  // last element: address of the end of the dual array
  eptr[ elem_nbr ] = 4*elem_nbr;

  // 
  // Metis partitioning
  int status  = 0;
  idx_t 
    // 1 - 2 elem share at least 1 vertex; 
    // 2 - 2 elem share at least 1 edge; 
    // 3 - 2 elem share at least 1 facet (triangl); ... 
    ncommon = 1,  /* Higher it is faster it is */
    nparts  = static_cast< idx_t >(n_partitions_); /*number of part*/
  idx_t objval;

  std::cout << "Metis partitioning" << std::endl;
  // 
  switch (METIS_GTYPE_DUAL/*params->gtype*/) 
    {
      //
    case METIS_GTYPE_DUAL:
      {
	status = METIS_PartMeshDual( &elem_nbr, &vertices_nbr, eptr, eind, NULL, NULL, 
				     &ncommon, &nparts, NULL, options, &objval, epart, npart );
	break;
      }
      // 
    case METIS_GTYPE_NODAL:
      {
	status = METIS_PartMeshNodal( &elem_nbr, &vertices_nbr, eptr, eind, NULL, NULL, 
				      &nparts, NULL, options, &objval, epart, npart );
	break;
      }
      // 
    default:
      {
	abort();
      }
    }
  // 
  switch ( status ) 
    {
    case METIS_OK://Indicates an input error
      {
	std::cout << "METIS partitioning successed" << std::endl;
	break;
      }
    case METIS_ERROR_INPUT://Indicates an input error
      {
	std::cerr << "input error" << std::endl;
	abort();
      }
    case METIS_ERROR_MEMORY://Indicates that it could not allocate the required memory.
      {
	std::cerr << "could not allocate the required memory" << std::endl;
	abort();
      }
    case METIS_ERROR://Indicates some other type of error
      {
	std::cerr << "ERROR" << std::endl;
	abort();
      }
    }
  
  // 
  // Copy the results
  std::copy( epart, epart + elem_nbr, elements_partitioning_.begin() ); 
  std::copy( npart, npart + vertices_nbr, nodes_partitioning_.begin() ); 

  // 
  // Free objects
  free(eptr);
  free(epart);
  free(eind);
  free(npart);

  //
  //
  Make_analysis();
}
//
//
//
void 
DPM::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
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
  std::string file_name("Parcellation_METIS_segment_");
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
DPM& 
DPM::operator = ( const DPM& that )
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
		       const DPM& that)
{

  //
  //
  return stream;
};
