#include "Electrodes_surface.h"

typedef Solver::PDE_solver_parameters SDEsp;

/* Electrodes_surface */
//
//
//
Solver::Electrodes_surface::Electrodes_surface( boost::shared_ptr< Solver::Electrodes_setup > Electrodes,
						const boost::shared_ptr< MeshFunction< std::size_t > > Boundaries,
						const std::map< std::size_t, std::size_t >& Map_Index_Cell ):
  SubDomain(), electrodes_(Electrodes), boundaries_(Boundaries)
{
  //
  // Map the mesh tetrahedron entities
  std::vector<MeshEntity> cell_entity( Boundaries->mesh()->num_cells() );
  //
  for ( MeshEntityIterator entity((*Boundaries->mesh()), /*dim cell = */ 3) ;
	!entity.end() ; ++entity )
    cell_entity[entity->index()] = *entity;

    
  //
  //  Sub domain iteration in electrodes volume
  auto it_facet_cell_index = Map_Index_Cell.begin();
  std::size_t 
    cell_index  = 0,
    facet_index = 0;
  bool        in_electrode = false;
  std::string electrode_label;
  //
  for (MeshEntityIterator facet( *(Boundaries->mesh()), Boundaries->dim() );
       !facet.end(); ++facet)
    if ( (*Boundaries)[*facet] == /*electrodes = */ 100 )
      {
	//
	facet_index = facet->index();
	// which electrode the facet belong to
	std::tie(electrode_label, in_electrode) = electrodes_->inside_probe( facet->midpoint() );
	//
	if( in_electrode )
	  {
	    // which cell belong the facet
	    it_facet_cell_index = Map_Index_Cell.find( facet_index );
	    if ( it_facet_cell_index ==  Map_Index_Cell.end() )
	      {
		std::cerr << "Error: no cell mapping for facet: " << facet_index << std::endl;
		abort();
	      }
	    //
	    cell_index = it_facet_cell_index->second;

	    //
	    // The facet midpoint is required for the boundary check
	    Point midpoint = facet->midpoint();
	    list_vertices_.push_back(std::make_tuple ( electrode_label,
						       midpoint,   // position
						       *facet,     // facet
						       cell_index, // cell index
						       -1,         // it is not a vertex
						       false ));
	    
	    //
	    // Vertices of the facet in electrod
//	    for (VertexIterator v( cell_entity[ cell_index ] ); !v.end(); ++v)
	    for (VertexIterator v( *facet ); !v.end(); ++v)
	      list_vertices_.push_back(std::make_tuple ( electrode_label, 
							 v->point(), // position
							 *facet,     // facet
							 cell_index, // cell index
							 v->index(), // it is a vertex
							 false ));
	  }
      }
}
//
//
//
void 
Solver::Electrodes_surface::surface_vertices_per_electrodes( const std::size_t Boundary_label )
{
  //
  std::map< std::string, std::map< std::size_t, std::list< MeshEntity  >  >  > map_boundary_cells;
  //
  for (MeshEntityIterator facet( *(boundaries_->mesh()), boundaries_->dim() );
       !facet.end(); ++facet)
    if ( (*boundaries_)[*facet] == /*electrodes = */  Boundary_label )
      {
	// The facet midpoint is required for the boundary check
	Point midpoint = facet->midpoint();
	for( auto it_vertex = list_vertices_.begin() ; it_vertex != list_vertices_.end() ;
	     it_vertex++ )
	  if( std::get< /*vertex index*/ 4 >(*it_vertex) == -1 )
	    if( std::get< /*position*/ 1 >(*it_vertex).distance( midpoint ) < 1.e-3 ) 
	      {
		map_boundary_cells[std::get< /*electrode label*/ 0 >(*it_vertex)]
		  [std::get< /*cell index*/ 3 >(*it_vertex)].push_back(std::get< /*facet*/ 2 >(*it_vertex));
		break;
	      }
      }

  //
  //
  electrodes_->set_boundary_cells( map_boundary_cells );


//  for ( auto electrode : map_boundary_cells )
//    for ( auto cell : electrode.second )
//      std::cout << electrode.first << " Cell index: " << cell.first << std::endl;

//  //
//  //
//  std::vector<int>  check(10, 0);
//  std::vector< std::set< std::size_t > >  tetrahedron( boundaries_->mesh()->num_cells() );
//  std::map< std::string/*electrode*/, 
//	    std::set< std::size_t > /*vertex index*/ > map_electrode_vertices;
//  std::map< std::string/*electrode*/, 
//	    std::set< std::size_t > /*cell index*/ > map_electrode_cell;
//
//  //
//  // List the cells with a vertex touching the boundaries
//  for ( auto vertex_101 : list_vertices_ )
//    if( std::get< /*vertex*/ 4 >(vertex_101)  && 
//	std::get< /*boundary*/ 5 >(vertex_101) )
//      {
//	int hit = (int)std::get< /* cell idx */ 3 >(vertex_101);
//	tetrahedron[hit].insert( std::get< /* vertex idx */ 2 >(vertex_101) );
//      }
//
//  //
//  // 
//  for ( int cell_101 = 0 ; cell_101 < tetrahedron.size() ; cell_101++ )
//    {
//      // Depending on the topography of the geometry, we can have 3 or 4 vertices of 
//      // a same tetrahedron on boundary.
//      // We will select only cells with three vertices on surface: num cells == num facets
//      if( tetrahedron[cell_101].size() == 3 || tetrahedron[cell_101].size() == 4 )
//	{
//	  //
//	  for ( auto vertex_101 : list_vertices_ )
//	    if ( std::get< /*cell idx*/ 3 >(vertex_101) ==  cell_101 && 
//		 std::get< /*vertex*/ 4 >(vertex_101) )
//	      {
//		map_electrode_cell[std::get<0>(vertex_101)].insert(std::get<3>(vertex_101));
//		map_electrode_vertices[std::get<0>(vertex_101)].insert(std::get<2>(vertex_101));
//	      }
//	}
//      check[ tetrahedron[cell_101].size() ] += 1;
//    }
//    
//  for ( auto test :  check )
//    std::cout << test << " ";
//  std::cout << std::endl;
//
//// //
//// //
//// int min_size = 1000000;
//// for (auto electrode : map_electrode_cell)
////   if( electrode.second.size() != 0 &&  electrode.second.size() < min_size )
////     min_size = electrode.second.size();
//// //
//// std::cout << " min_ size: " << min_size << std::endl;
//// min_size -= 100;
//// std::cout << " min_ size - 100: " << min_size << std::endl;
//// //
//// for ( auto electrode = map_electrode_cell.begin() ; 
////       electrode != map_electrode_cell.end() ;
////	electrode++ )
////   {
////     auto cells_it = electrode->second.begin();
////      electrode->second.erase( std::next(cells_it, 3), electrode->second.end() );
////   } 
//
//  
//
//
//  for (auto electrode : map_electrode_vertices)
//    std::cout << electrode.first << ": " << electrode.second.size() << std::endl;
//  //
//  for (auto electrode : map_electrode_cell)
//    for (auto cells : electrode.second )
//      std::cout << electrode.first << " Cell index: " << cells << std::endl;
//
//
////  for (auto electrode : map_electrode_cell)
////    {
////      std::cout << electrode.first << ": " << electrode.second.size() << std::endl;
////      for (auto cell : electrode.second )
////	std::cout << "  " << cell << std::endl;
////    }
//  
//  //
//  //
//  electrodes_->set_boundary_cells( map_electrode_cell );
//  electrodes_->set_boundary_vertices( map_electrode_vertices );
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
