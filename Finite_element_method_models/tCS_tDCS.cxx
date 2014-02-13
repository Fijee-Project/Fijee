#include <iostream>
#include "tCS_tDCS.h"

typedef Solver::PDE_solver_parameters SDEsp;

//#include <tuple>        // std::tuple, std::get, std::tie, std::ignore
//class ElectrodesSurface : public SubDomain
//{
//private:
//  //!
//  boost::shared_ptr< Solver::Electrodes_setup > electrodes_;
//  boost::shared_ptr< MeshFunction< std::size_t > > boundaries_;
//  std::list<Point> list_electrode_surface_;
////  mutable std::list< std::tuple< Point,        // vertex (can't take vertex directly: no operator =)
////				 std::size_t,  // vertex id (mesh 2D)
////				 std::size_t,  // cell id (facet)
////				 bool,         // vertex: true ; midpoint: false
////				 bool,         // satisfaction criteria
////				 std::string > // electrode label
////		     > list_facet_vertex_;
//  mutable std::map< std::string, std::tuple< std::string, // electrode label
//				             Point,        // vertex (can't take vertex directly: no operator =)
//					     std::size_t,  // vertex index (mesh)
//					     std::size_t,  // cell index (tetrahedron)
//					     bool,         // vertex: true ; midpoint: false
//					     bool>         // criteria satisfaction
//		    > map_vertices_;
//  mutable std::list< std::tuple< std::string, // electrode label
//				 Point,        // vertex (can't take vertex directly: no operator =)
//				 std::size_t,  // vertex index (mesh)
//				 std::size_t,  // cell index (tetrahedron)
//				 bool,         // vertex: true ; midpoint: false
//				 bool>         // criteria satisfaction
//		    > list_vertices_;
//
//public:
//  ElectrodesSurface():SubDomain(){};
//  ElectrodesSurface( const boost::shared_ptr< Solver::Electrodes_setup > Electrodes,
//		     const boost::shared_ptr< MeshFunction< std::size_t > > Boundaries,
//		     const std::map< std::size_t, std::size_t >& Map_Index_Cell ):
//    SubDomain(),
//    electrodes_(Electrodes), boundaries_(Boundaries)
//  {
//    // 
//    //
////    for( SubsetIterator facet(*Boundaries,  100) ; !facet.end() ; ++facet )
////      {
//////	std::cout /*<< facet->mesh_id() << " " */
//////		  << facet->index() << " " 
//////		  << (*Boundaries)[*facet] << " " 
//////	  /*<< facet->midpoint() << " " 
//////	    << facet->midpoint().norm() */<< std::endl;
//////
//////	int num_vertices = 0;
//////	for (VertexIterator v(*facet); !v.end(); ++v)
//////	  {
//////	    std::cout << "vertex " << num_vertices++ << ": " << v->point() << std::endl;
//////	  }
////	list_electrode_surface_.push_back( facet->midpoint() );
////
////	
////	int num_vertices = 0;
////	
////	if ( electrodes_->inside(facet->midpoint()) )
////	  {
////	    
//////	    map_vertices_[std::to_string( facet->midpoint().dot(facet->midpoint()) )] = 
//////	      std::make_tuple ( facet->midpoint(), 
//////				facet->index(),
//////				0,
//////				false);
////	    list_facet_vertex_.push_back( std::make_tuple ( facet->midpoint(),
////							    0,
////							    facet->index(),
////							    false,
////							    false,
////							    "") );
////	    
////	    //
////	    for (VertexIterator v(*facet); !v.end(); ++v)
////	      {
//////		std::cout << "vertex " << num_vertices++ << ": " << v->point() << " " << v->index() << std::endl;
//////		map_vertices_[std::to_string( v->point().dot(v->point()) )] = 
//////		  std::make_tuple ( v->point(), 
//////				    facet->index(),
//////				    v->index(),
//////				    false);
////		list_facet_vertex_.push_back( std::make_tuple ( v->point(),
////								v->index(),
////								facet->index(),
////								true,
////								false,
////								"") );
////	      }
////	  }
////      } 
//
//    //
//    // 
//    //
//
//    //
//    // Map the mesh tetrahedron entities
//    std::vector<MeshEntity> cell_entity( Boundaries->mesh()->num_cells() );
//    //
//    for ( MeshEntityIterator entity((*Boundaries->mesh()), /*dim cell = */3) ;
//	  !entity.end() ; ++entity )
//      cell_entity[entity->index()] = *entity;
//
//    
//    //
//    //  Sub domain iteration in electrodes volume
//    auto it_facet_cell_index = Map_Index_Cell.begin();
//    std::size_t cell_index = 0.;
//    bool        in_electrode = false;
//    std::string electrode_label;
//    //
//    for (MeshEntityIterator facet( *(Boundaries->mesh()), Boundaries->dim() );
//	 !facet.end(); ++facet)
//      {
//	if ( (*Boundaries)[*facet] == 100 )
//	  {
//	    //
//	    std::tie(electrode_label, in_electrode) = electrodes_->inside_probe( facet->midpoint() );
//	    if( in_electrode )
//	      {
//		// which cell belong the facet
//		it_facet_cell_index = Map_Index_Cell.find( facet->index() );
//		if ( it_facet_cell_index ==  Map_Index_Cell.end() )
//		  {
//		    std::cerr << "Error: no cell mapping for facet: " << facet->index() << std::endl;
//		    abort();
//		  }
//		//
//		cell_index = it_facet_cell_index->second;
//		
////	    std::cout << facet->mesh_id() << " " 
////		      << facet->index() << " " 
////	      //	      << facet->global_index() << " " 
////		      <<  cell_index << " " 
////		      << (*Boundaries)[*facet] << " " 
////	      /*<< facet->midpoint() << " " 
////		<< facet->midpoint().norm() */<< std::endl;
//
//		// The facet midpoint is required for the boundary check
//		Point midpoint = facet->midpoint();
//		list_vertices_.push_back(std::make_tuple ( electrode_label,
//							   midpoint, 
//							   -1,
//							   cell_index,
//							   false, 
//							   false ));
//	    
//	    
//	    
//////		int num_vertices = 0;
////		for (VertexIterator v(*facet); !v.end(); ++v)
////		  {
//////		    std::cout << "vertex facet " << num_vertices++ << ": " 
//////			      << v->point() << " " << v->index() << std::endl;
////		    //
////		    list_vertices_.push_back(std::make_tuple ( electrode_label,
////							       v->point(), 
////							       v->index(),
////							       cell_index,
////							       true, 
////							       false ));
////		    
////		  }
////		num_vertices = 0;
//		for (VertexIterator v( cell_entity[ cell_index ] ); !v.end(); ++v)
//		  {
////		    std::cout << "vertex tetra " << num_vertices++ << ": " << v->point() << " " << v->index() << std::endl;
//		    list_vertices_.push_back(std::make_tuple ( electrode_label,
//							       v->point(), 
//							       v->index(),
//							       cell_index,
//							       true, 
//							       false ));
//		  }
////	    list_electrode_surface_.push_back( facet->midpoint() );
//	      }
//	  }
//      }
//  };
//
//public:
//  /*!
//   *
//   *
//   * 
//   *
//   */
//  int num_hit()
//  {
//    //
//    //
//    std::vector<int>  check(10, 0);
//    std::vector< std::set< std::size_t > >  tetrahedron( boundaries_->mesh()->num_cells() );
//    std::vector<int>  facet_hit(boundaries_->size(), 0);
//    std::vector<bool> facet_on(boundaries_->size(), false);
//    std::map< std::string, std::set< std::size_t > > map_electrode_cell_vertices;
//
//    //
//    // List the cells with a vertex touching the boundaries
//    for ( auto vertex_101 : list_vertices_ )
//      if( /*vertex*/std::get<4>(vertex_101)  && 
//	  /*boundary*/std::get<5>(vertex_101) )
//	{
////	  std::cout 
////	    << std::get<0>(vertex_101) << " "
////	    << std::get<2>(vertex_101) << " "
////	    << std::get<3>(vertex_101) << " "
////	    << std::get<4>(vertex_101) << " "
////	    << std::get<5>(vertex_101) << "\n"
////	    << std::get<1>(vertex_101) 
////	    << std::endl;
//	  int hit = (int)std::get<3>(vertex_101);
//	  tetrahedron[hit].insert( std::get<2>(vertex_101) );
//	}
//
//
//    //
//    // 
//    for ( int cell_101 = 0 ; cell_101 < tetrahedron.size() ; cell_101++ )
//      {
//	// Depending on the topography of the geometry, we can have 3 or 4 vertices of 
//	// a same tetrahedron on boundary
//	if( tetrahedron[cell_101].size() == 3 || tetrahedron[cell_101].size() == 4)
//	  {
//	    //
//	    for ( auto vertex_101 : list_vertices_ )
//	      if ( std::get<3>(vertex_101) ==  cell_101 && std::get<4>(vertex_101) )
//		{
//		  map_electrode_cell_vertices[std::get<0>(vertex_101)].insert(std::get<2>(vertex_101));
//		  //		  std::cout << std::get<1>(vertex_101) << std::endl;
//		}
//	  }
//	check[ tetrahedron[cell_101].size() ] += 1;
//      }
//
////   //
////   int num_hit = 0;
////   for ( int facet_101 = 0 ; facet_101 < facet_hit.size() ; facet_101++ )
////     {
////	if( facet_hit[facet_101] == 3 )
////	  {
////	    //	    facet_on[facet_101] = true;
////	    for ( auto vertex_101 : list_facet_vertex_ )
////	      if( std::get<2>(vertex_101) ==  facet_101 &&  std::get<3>(vertex_101) )
////		{
////		  map_vertex_electrodes[std::get<1>(vertex_101)] = std::get<5>(vertex_101);
////		  ++num_hit;
////		  std::cout << std::get<0>(vertex_101) << std::endl;
////		}
////	  }
////	check[ facet_hit[facet_101] ] += 1;
////     }
//
//    for ( auto test :  check )
//      std::cout << test << " ";
//    std::cout << std::endl;
//
//    for (auto electrode : map_electrode_cell_vertices)
//      std::cout << electrode.first << ": " << electrode.second.size() << std::endl;
// 
//    //
//    //
//    return 0.;
//  };
//
//private:
//  virtual bool inside(const Array<double>& x, bool on_boundary) const
//  {
////    //
////    //
//    Point vertex_point( x[0], x[1], x[2]);
//    //    std::string vertex_finger_print = std::to_string( vertex_point.dot(vertex_point) );
//    bool on_electrode = false;
//
////    if(on_boundary)
////      for ( auto point : list_electrode_surface_ )
////	if ( point.squared_distance(vertex_point) < 1. )
////	  {
////	    on_electrode = electrodes_->inside( vertex_point );
////	  }
//    
//
////    //
////    if(on_boundary)
////      if( electrodes_->inside( vertex_point ) )
////	{
////	  //
////	  auto it_vertex = map_vertices_.find( vertex_finger_print );
////	  //
////	  if ( it_vertex != map_vertices_.end() )
////	    // second check on point
////	    if( (std::get<1>(it_vertex->second)).distance( vertex_point ) < 1.e-3 ) 
////	      {
////		// Satisfaction criteria fulfilled 
////		std::get<5>(it_vertex->second) = true;
////		on_electrode                   = true;
////	      }
////	}
//
//    //
//    if( on_boundary )
//      if( electrodes_->inside( vertex_point ) )
//	{
//	  for( auto it_vertex = list_vertices_.begin() ; it_vertex != list_vertices_.end() ;
//	       it_vertex++ )
//	    if( std::get<1>(*it_vertex).distance( vertex_point ) < 1.e-3 ) 
//	      {
//		// Satisfaction criteria fulfilled 
//		std::get<5>(*it_vertex) = true;
//		on_electrode            = true;
//	      }
//	}   
//
////    if(on_boundary)
////      {
////	//
////	bool in_electrode = false;
////	std::string electrode_label;
////	//
////	std::tie(electrode_label, in_electrode) = electrodes_->inside_probe( vertex_point );
////	if( in_electrode )
////	  {
////	    for ( auto vertex_tuple = list_facet_vertex_.begin() ; 
////		  vertex_tuple != list_facet_vertex_.end() ; 
////		  vertex_tuple++)
////	      if( (std::get<0>(*vertex_tuple)).distance(vertex_point) < 1.e-3 )
////		{
////		  std::get<4>(*vertex_tuple) = true;
////		  std::get<5>(*vertex_tuple) = electrode_label;
////		  on_electrode = true;
////		}
////	  }
////      }
//
////   if (on_electrode)
////     std::cout << list_vertices_.size() << std::endl;
//
////    if(on_boundary)
////      if( electrodes_->inside( vertex_point ) )
////	on_electrode = true;
//    
//
//
//    //
//    
//    //
//    //
//    return ( on_electrode );
//  }
//};


//
//
//
Solver::tCS_tDCS::tCS_tDCS()
{
  //
  // Load the mesh
  std::cout << "Load the mesh" << std::endl;
  //
  std::string mesh_xml = (SDEsp::get_instance())->get_files_path_output_();
  mesh_xml += "mesh.xml";
  //
  mesh_.reset( new Mesh(mesh_xml.c_str()) );
  //
  info( *mesh_ );

  //
  // Load Sub_domains
  std::cout << "Load Sub_domains" << std::endl;
  //
  std::string subdomains_xml = (SDEsp::get_instance())->get_files_path_output_();
  subdomains_xml += "mesh_subdomains.xml";
  //
  domains_.reset( new MeshFunction< long unsigned int >(mesh_, subdomains_xml.c_str()) );
  // write domains
  std::string domains_file_name = (SDEsp::get_instance())->get_files_path_result_();
  domains_file_name            += std::string("domains.pvd");
  File domains_file( domains_file_name.c_str() );
  domains_file << *domains_;

  //
  // Load the conductivity. Anisotrope conductivity
  std::cout << "Load the conductivity" << std::endl;
  sigma_.reset( new Solver::Tensor_conductivity(mesh_) );

  //
  // Define the function space
  V_.reset( new tCS_model::FunctionSpace(mesh_) );

  //
  // Read the electrodes xml file
  electrodes_.reset( new Electrodes_setup() );



  //
  // Boundary marking
  //
  
  //
  // Boundary conditions
  std::cout << "Load boundaries" << std::endl;

  //
  // Load the facets collection
  std::cout << "Load facets collection" << std::endl;
  std::string facets_collection_xml = (SDEsp::get_instance())->get_files_path_output_();
  facets_collection_xml += "mesh_facets_subdomains.xml";
  //
  mesh_facets_collection_.reset( new MeshValueCollection< std::size_t > (*mesh_, facets_collection_xml) );

  //
  // MeshDataCollection methode
  // Recreate the connectivity
  // Get mesh connectivity D --> d
  const std::size_t d = mesh_facets_collection_->dim();
  const std::size_t D = mesh_->topology().dim();
  dolfin_assert(d == 2);
  dolfin_assert(D == 3);

  //
  // Generate connectivity if it does not excist
  mesh_->init(D, d);
  const MeshConnectivity& connectivity = mesh_->topology()(D, d);
  dolfin_assert(!connectivity.empty());
  
  //
  // Map the facet index with cell index
  std::map< std::size_t, std::size_t > map_index_cell;
  typename std::map<std::pair<std::size_t, std::size_t>, std::size_t>::const_iterator it;
  const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& values
    = mesh_facets_collection_->values();
  // Iterate over all values
  for ( it = values.begin() ; it != values.end() ; ++it )
    {
      // Get value collection entry data
      const std::size_t cell_index = it->first.first;
      const std::size_t local_entity = it->first.second;
      const std::size_t value = it->second;

      std::size_t entity_index = 0;
      // Get global (local to to process) entity index
      dolfin_assert(cell_index < _mesh->num_cells());
      map_index_cell[connectivity(cell_index)[local_entity]] = cell_index;
 
      // Set value for entity
      dolfin_assert(entity_index < _size);
    }

  
  //
  // Define boundary condition
  boundaries_.reset( new MeshFunction< std::size_t >(mesh_) );
  *boundaries_ = *mesh_facets_collection_;
  //
  boundaries_->rename( mesh_facets_collection_->name(),
		       mesh_facets_collection_->label() );

  //
  // Boundary definition
  Electrodes_surface electrodes_101( electrodes_, boundaries_, map_index_cell );
  //
  electrodes_101.mark(*boundaries_, 101);
  electrodes_101.surface_vertices_per_electrodes();
  // write boundaries
  std::string boundaries_file_name = (SDEsp::get_instance())->get_files_path_result_();
  boundaries_file_name            += std::string("boundaries.pvd");
  File boundaries_file( boundaries_file_name.c_str() );
  boundaries_file << *boundaries_;
}
//
//
//
void 
Solver::tCS_tDCS::operator () ( /*Solver::Phi& source,
				  SLD_model::FunctionSpace& V,
				  FacetFunction< size_t >& boundaries*/)
{
//  //
//  // Mutex the electrodes vector poping process
//  //
//  Solver::Current_density source;
//    try {
//      // lock the electrode list
//      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
//      source = electrodes_list_.front();
//      electrodes_list_.pop_front();
//    }
//    catch (std::logic_error&) {
//      std::cout << "[exception caught]\n";
//    }
//
//  //
//  //
//  std::cout << source.get_name_() << std::endl;
//  
////  //
////  // Define Dirichlet boundary conditions 
////  DirichletBC boundary_conditions(*V, source, perifery);


  //////////////////////////////////////////////////////
  // Transcranial direct current stimulation equation //
  //////////////////////////////////////////////////////
      



//  //
//  // PDE boundary conditions
//  DirichletBC bc(*V_, *(electrodes_->get_potential()), *boundaries_, 101);

  //
  // Define variational forms
  tCS_model::BilinearForm a(V_, V_);
  tCS_model::LinearForm L(V_);
      
  //
  // Anisotropy
  // Bilinear
  a.a_sigma  = *sigma_;
  // a.dx       = *domains_;
  
  
  // Linear
  L.I  = *(electrodes_->get_current());
  //  L.ds = *boundaries_;
  // L.Se = Se;

  //
  // Compute solution
  Function u(*V_);
  LinearVariationalProblem problem(a, L, u/*, bc*/);
  LinearVariationalSolver  solver(problem);
  // krylov
  solver.parameters["linear_solver"]  
    = (SDEsp::get_instance())->get_linear_solver_();
  solver.parameters("krylov_solver")["maximum_iterations"] 
    = (SDEsp::get_instance())->get_maximum_iterations_();
  solver.parameters("krylov_solver")["relative_tolerance"] 
    = (SDEsp::get_instance())->get_relative_tolerance_();
  solver.parameters["preconditioner"] 
    = (SDEsp::get_instance())->get_preconditioner_();
  //
  solver.solve();

 //
 // Regulation terme
 //  \int u dx = 0
 double old_u_bar = 0.;
 double u_bar = 1.e+6;
 double U_bar = 0.;
 double N = u.vector()->size();
 int iteration = 0;
 double Sum = 1.e+6;
 //
 //  while ( abs( u_bar - old_u_bar ) > 0.1 )
 while ( abs(Sum) > 1.e-3 )
   {
     old_u_bar = u_bar;
     u_bar  = u.vector()->sum();
     u_bar /= N;
     (*u.vector()) -= u_bar;
     //
     U_bar += u_bar;
     Sum = u.vector()->sum();
     std::cout << ++iteration << " ~ " << Sum  << std::endl;
   }
 
 std::cout << "int u dx = " << Sum << " " << U_bar << std::endl;
  
//  std::list<std::size_t> test;
//  regulation_factor(u, test);


  //
  // Filter function over a subdomain
  std::list<std::size_t> sub_domains{4,5};
  solution_domain_extraction(u, sub_domains);


//  //
//  // Extract subfunctions
//  Function potential = u[0];

  //
  // Save solution in VTK format
  //  * Binary (.bin)
  //  * RAW    (.raw)
  //  * SVG    (.svg)
  //  * XD3    (.xd3)
  //  * XDMF   (.xdmf)
  //  * XML    (.xml) // FEniCS xml
  //  * XYZ    (.xyz)
  //  * VTK    (.pvd) // paraview
  std::string file_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS.pvd");
  File file( file_name.c_str() );
  //
  file << u;
};
//
//
//
void
Solver::tCS_tDCS::solution_domain_extraction(const Function& u, std::list<std::size_t>& Sub_domains)
{
  // 
  const std::size_t num_vertices = mesh_->num_vertices();
  
  // Get number of components
  const std::size_t dim = u.value_size();
  
  // Open file
  std::string sub_dom("tDCS");
  for ( auto sub : Sub_domains )
    sub_dom += std::string("_") + std::to_string(sub);
  sub_dom += std::string(".vtu");
  //
  std::string extracted_solution = (SDEsp::get_instance())->get_files_path_result_();
  extracted_solution            += sub_dom;
  //
  std::ofstream VTU_xml_file(extracted_solution);
  VTU_xml_file.precision(16);

  // Allocate memory for function values at vertices
  const std::size_t size = num_vertices * dim; // dim = 1
  std::vector<double> values(size);
  u.compute_vertex_values(values, *mesh_);
 
  //
  // 
  std::vector<int> V(num_vertices, -1);


  //
  //
  int 
    num_tetrahedra = 0,
    offset = 0,
    inum = 0;
  //
  std::string 
    vertices_position_string,
    vertices_associated_to_tetra_string,
    offsets_string,
    cells_type_string,
    point_data;
  // loop over mesh cells
  for ( CellIterator cell(*mesh_) ; !cell.end() ; ++cell )
    // loop over extraction sub-domains
    for( auto sub_domain : Sub_domains ) 
      if ( (*domains_)[cell->index()] == sub_domain )
	{
	  //  vertex id
	  for ( VertexIterator vertex(*cell) ; !vertex.end() ; ++vertex )
	    {
	      if( V[ vertex->index() ] == -1 )
		{
		  //
		  V[ vertex->index() ] = inum++;
		  vertices_position_string += 
		    std::to_string( vertex->point().x() ) + " " + 
		    std::to_string( vertex->point().y() ) + " " +
		    std::to_string( vertex->point().z() ) + " " ;
		  point_data += std::to_string( values[vertex->index()] ) + " ";
		}

	      //
	      // Volume associated
	      vertices_associated_to_tetra_string += 
		std::to_string( V[vertex->index()] ) + " " ;
	    }
      
	  //
	  // Offset for each volumes
	  offset += 4;
	  offsets_string += std::to_string( offset ) + " ";
	  //
	  cells_type_string += "10 ";
	  //
	  num_tetrahedra++;
	}

  //
  // header
  VTU_xml_file << "<?xml version=\"1.0\"?>" << std::endl;
  VTU_xml_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
  VTU_xml_file << "  <UnstructuredGrid>" << std::endl;

  // 
  // vertices and values
  VTU_xml_file << "    <Piece NumberOfPoints=\"" << inum
	       << "\" NumberOfCells=\"" << num_tetrahedra << "\">" << std::endl;
  VTU_xml_file << "      <Points>" << std::endl;
  VTU_xml_file << "        <DataArray type = \"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  VTU_xml_file << vertices_position_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "      </Points>" << std::endl;
  
  //
  // Point data
  VTU_xml_file << "      <PointData Scalars=\"scalars\">" << std::endl;
  VTU_xml_file << "        <DataArray type=\"Float32\" Name=\"scalars\" format=\"ascii\">" << std::endl; 
  VTU_xml_file << point_data << std::endl; 
  VTU_xml_file << "         </DataArray>" << std::endl; 
  VTU_xml_file << "      </PointData>" << std::endl; 
 
  //
  // Tetrahedra
  VTU_xml_file << "      <Cells>" << std::endl;
  VTU_xml_file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
  VTU_xml_file << vertices_associated_to_tetra_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
  VTU_xml_file << offsets_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
  VTU_xml_file << cells_type_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "      </Cells>" << std::endl;
  VTU_xml_file << "    </Piece>" << std::endl;

  //
  // Tail
  VTU_xml_file << "  </UnstructuredGrid>" << std::endl;
  VTU_xml_file << "</VTKFile>" << std::endl;
}
//
//
//
void
Solver::tCS_tDCS::regulation_factor(const Function& u, std::list<std::size_t>& Sub_domains)
{
  // 
  const std::size_t num_vertices = mesh_->num_vertices();
  
  // Get number of components
  const std::size_t dim = u.value_size();
  
  // Open file
  std::string sub_dom("tDCS");
  for ( auto sub : Sub_domains )
    sub_dom += std::string("_") + std::to_string(sub);
  sub_dom += std::string("_bis.vtu");
  //
  std::string extracted_solution = (SDEsp::get_instance())->get_files_path_result_();
  extracted_solution            += sub_dom;
  //
  std::ofstream VTU_xml_file(extracted_solution);
  VTU_xml_file.precision(16);

  // Allocate memory for function values at vertices
  const std::size_t size = num_vertices * dim; // dim = 1
  std::vector<double> values(size);
  u.compute_vertex_values(values, *mesh_);
 
  //
  // 
  std::vector<int> V(num_vertices, -1);

  //
  // int u dx = 0
  double old_u_bar = 0.;
  double u_bar = 1.e+6;
  double U_bar = 0.;
  double N = size;
  int iteration = 0;
  double Sum = 1.e+6;
  //
  //  while ( abs( u_bar - old_u_bar ) > 0.1 )
  while ( abs(Sum) > .01 || abs((old_u_bar - u_bar) / old_u_bar) > 1. /* % */ )
    {
      old_u_bar = u_bar;
      u_bar = 0.;
      for ( double val : values ) u_bar += val;
      u_bar /= N;
      std::for_each(values.begin(), values.end(), [&u_bar](double& val){val -= u_bar;});
      //
      U_bar += u_bar;
      Sum = 0;
      for ( double val : values ) Sum += val;
      std::cout << ++iteration << " ~ " << Sum  << " ~ " << u_bar << std::endl;
    }

  std::cout << "int u dx = " << Sum << " " << U_bar << std::endl;
  std::cout << "Size = " << Sub_domains.size()  << std::endl;
 

  //
  //
  int 
    num_tetrahedra = 0,
    offset = 0,
    inum = 0;
  //
  std::string 
    vertices_position_string,
    vertices_associated_to_tetra_string,
    offsets_string,
    cells_type_string,
    point_data;
  // loop over mesh cells
  for ( CellIterator cell(*mesh_) ; !cell.end() ; ++cell )
    // loop over extraction sub-domains
//    for( auto sub_domain : Sub_domains ) 
//     if ( (*domains_)[cell->index()] == sub_domain || Sub_domains.size() == 0 )
	{
	  //  vertex id
	  for ( VertexIterator vertex(*cell) ; !vertex.end() ; ++vertex )
	    {
	      if( V[ vertex->index() ] == -1 )
		{
		  //
		  V[ vertex->index() ] = inum++;
		  vertices_position_string += 
		    std::to_string( vertex->point().x() ) + " " + 
		    std::to_string( vertex->point().y() ) + " " +
		    std::to_string( vertex->point().z() ) + " " ;
		  point_data += std::to_string( values[vertex->index()] ) + " ";
		}

	      //
	      // Volume associated
	      vertices_associated_to_tetra_string += 
		std::to_string( V[vertex->index()] ) + " " ;
	    }
      
	  //
	  // Offset for each volumes
	  offset += 4;
	  offsets_string += std::to_string( offset ) + " ";
	  //
	  cells_type_string += "10 ";
	  //
	  num_tetrahedra++;
	}

  //
  // header
  VTU_xml_file << "<?xml version=\"1.0\"?>" << std::endl;
  VTU_xml_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
  VTU_xml_file << "  <UnstructuredGrid>" << std::endl;

  // 
  // vertices and values
  VTU_xml_file << "    <Piece NumberOfPoints=\"" << inum
	       << "\" NumberOfCells=\"" << num_tetrahedra << "\">" << std::endl;
  VTU_xml_file << "      <Points>" << std::endl;
  VTU_xml_file << "        <DataArray type = \"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  VTU_xml_file << vertices_position_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "      </Points>" << std::endl;
  
  //
  // Point data
  VTU_xml_file << "      <PointData Scalars=\"scalars\">" << std::endl;
  VTU_xml_file << "        <DataArray type=\"Float32\" Name=\"scalars\" format=\"ascii\">" << std::endl; 
  VTU_xml_file << point_data << std::endl; 
  VTU_xml_file << "         </DataArray>" << std::endl; 
  VTU_xml_file << "      </PointData>" << std::endl; 
 
  //
  // Tetrahedra
  VTU_xml_file << "      <Cells>" << std::endl;
  VTU_xml_file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
  VTU_xml_file << vertices_associated_to_tetra_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
  VTU_xml_file << offsets_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
  VTU_xml_file << cells_type_string << std::endl;
  VTU_xml_file << "        </DataArray>" << std::endl;
  VTU_xml_file << "      </Cells>" << std::endl;
  VTU_xml_file << "    </Piece>" << std::endl;

  //
  // Tail
  VTU_xml_file << "  </UnstructuredGrid>" << std::endl;
  VTU_xml_file << "</VTKFile>" << std::endl;
}
