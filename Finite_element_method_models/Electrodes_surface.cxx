#include "Electrodes_surface.h"

typedef Solver::PDE_solver_parameters SDEsp;

/* Electrodes_surface */
//
//
//
Solver::Electrodes_surface::Electrodes_surface( const boost::shared_ptr< Solver::Electrodes_setup > Electrodes,
						const boost::shared_ptr< MeshFunction< std::size_t > > Boundaries,
						const std::map< std::size_t, std::size_t >& Map_Index_Cell ):
  SubDomain(), electrodes_(Electrodes), boundaries_(Boundaries)
{
  //
  // Map the mesh tetrahedron entities
  std::vector<MeshEntity> cell_entity( Boundaries->mesh()->num_cells() );
  //
  for ( MeshEntityIterator entity((*Boundaries->mesh()), /*dim cell = */3) ;
	!entity.end() ; ++entity )
    cell_entity[entity->index()] = *entity;

    
  //
  //  Sub domain iteration in electrodes volume
  auto it_facet_cell_index = Map_Index_Cell.begin();
  std::size_t cell_index = 0.;
  bool        in_electrode = false;
  std::string electrode_label;
  //
  for (MeshEntityIterator facet( *(Boundaries->mesh()), Boundaries->dim() );
       !facet.end(); ++facet)
    if ( (*Boundaries)[*facet] == /*electrodes = */ 100 )
      {
	// which electrode the facet belong to
	std::tie(electrode_label, in_electrode) = electrodes_->inside_probe( facet->midpoint() );
	if( in_electrode )
	  {
	    // which cell belong the facet
	    it_facet_cell_index = Map_Index_Cell.find( facet->index() );
	    if ( it_facet_cell_index ==  Map_Index_Cell.end() )
	      {
		std::cerr << "Error: no cell mapping for facet: " << facet->index() << std::endl;
		abort();
	      }
	    //
	    cell_index = it_facet_cell_index->second;

	    //
	    // The facet midpoint is required for the boundary check
	    Point midpoint = facet->midpoint();
	    list_vertices_.push_back(std::make_tuple ( electrode_label,
						       midpoint,   // position
						       -1,         // no regular index
						       cell_index,
						       false,      // it is not a vertex
						       false ));
	    
	    //
	    // Vertices of the cell the facet belong to
	    for (VertexIterator v( cell_entity[ cell_index ] ); !v.end(); ++v)
	      list_vertices_.push_back(std::make_tuple ( electrode_label,
							 v->point(), 
							 v->index(),
							 cell_index,
							 true, 
							 false ));
	  }
      }
}
//
//
//
void 
Solver::Electrodes_surface::surface_vertices_per_electrodes()
{
  //
  //
  std::vector<int>  check(10, 0);
  std::vector< std::set< std::size_t > >  tetrahedron( boundaries_->mesh()->num_cells() );
  std::map< std::string, std::set< std::size_t > > map_electrode_cell_vertices;
  std::map< std::string, std::set< std::size_t > > map_electrode_cell;

  //
  // List the cells with a vertex touching the boundaries
  for ( auto vertex_101 : list_vertices_ )
    if( std::get< /*vertex*/ 4 >(vertex_101)  && 
	std::get< /*boundary*/ 5 >(vertex_101) )
      {
	int hit = (int)std::get< /*cell idx*/ 3 >(vertex_101);
	tetrahedron[hit].insert( std::get< /*vertex idx*/ 2 >(vertex_101) );
      }


  //
  // 
  for ( int cell_101 = 0 ; cell_101 < tetrahedron.size() ; cell_101++ )
    {
      // Depending on the topography of the geometry, we can have 3 or 4 vertices of 
      // a same tetrahedron on boundary
      if( tetrahedron[cell_101].size() == 3 || tetrahedron[cell_101].size() == 4)
	{
	  //
	  for ( auto vertex_101 : list_vertices_ )
	    if ( std::get< /*cell idx*/ 3 >(vertex_101) ==  cell_101 && 
		 std::get< /*vertex*/ 4 >(vertex_101) )
	      {
		map_electrode_cell[std::get<0>(vertex_101)].insert(std::get<3>(vertex_101));
		map_electrode_cell_vertices[std::get<0>(vertex_101)].insert(std::get<2>(vertex_101));
	      }
	}
      check[ tetrahedron[cell_101].size() ] += 1;
    }
    
  for ( auto test :  check )
    std::cout << test << " ";
  std::cout << std::endl;

  for (auto electrode : map_electrode_cell_vertices)
    std::cout << electrode.first << ": " << electrode.second.size() << std::endl;
//  for (auto electrode : map_electrode_cell)
//    {
//      std::cout << electrode.first << ": " << electrode.second.size() << std::endl;
//      for (auto cell : electrode.second )
//	std::cout << "  " << cell << std::endl;
//    }
  
  //
  //
  electrodes_->set_boundary_cells( map_electrode_cell );
}
//
//
//
bool 
Solver::Electrodes_surface::inside(const Array<double>& x, bool on_boundary) const
{
  //
  Point vertex_point( x[0], x[1], x[2]);
  bool on_electrode = false;

  //
  //
  if( on_boundary )
    if( electrodes_->inside( vertex_point ) )
      for( auto it_vertex = list_vertices_.begin() ; it_vertex != list_vertices_.end() ;
	   it_vertex++ )
	if( std::get< /*position*/ 1 >(*it_vertex).distance( vertex_point ) < 1.e-3 ) 
	  {
	    // Satisfaction criteria fulfilled 
	    std::get< /*criteria*/ 5 >(*it_vertex) = true;
	    on_electrode            = true;
	  }
  
  //
  //
  return ( on_electrode );
}
