#include <iostream>
#include "tCS_tDCS_local_conductivity.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::tCS_tDCS_local_conductivity::tCS_tDCS_local_conductivity():Physics()
{
  //
  // Define the function space
  V_.reset( new tCS_model::FunctionSpace(mesh_) );
  V_field_.reset( new tCS_field_model::FunctionSpace(mesh_) );

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
  std::cout << "Load facets collection file" << std::endl;
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
      //      dolfin_assert(cell_index < mesh_->num_cells());
      map_index_cell[connectivity(cell_index)[local_entity]] = cell_index;
 
      // Set value for entity
      //  dolfin_assert(entity_index < _size);
    }

  
  //
  // Define boundary condition
  boundaries_.reset( new MeshFunction< std::size_t >(mesh_) );
  *boundaries_ = *mesh_facets_collection_;
  //
  boundaries_->rename( mesh_facets_collection_->name(),
		       mesh_facets_collection_->label() );
  //
  mesh_facets_collection_.reset();

  //
  // Boundary definition
  Electrodes_surface electrodes_101( electrodes_, boundaries_, map_index_cell );
  //
  electrodes_101.mark( *boundaries_, 101 );
  electrodes_101.surface_vertices_per_electrodes( 101 );
  // write boundaries
  std::string boundaries_file_name = (SDEsp::get_instance())->get_files_path_result_();
  boundaries_file_name            += std::string("boundaries.pvd");
  File boundaries_file( boundaries_file_name.c_str() );
  boundaries_file << *boundaries_;

  //
  // Local conductivity estimation - initialization
  // 
  
  // 
  // Limites of conductivities
  conductivity_boundaries_[OUTSIDE_SCALP]   = std::make_tuple(0.005, 1.);
  conductivity_boundaries_[OUTSIDE_SKULL]   = std::make_tuple(4.33e-03, 6.86e-03);
  conductivity_boundaries_[SPONGIOSA_SKULL] = std::make_tuple(5.66e-03, 23.2e-03);
  // 
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> uniform_dist_skin(0.005, 1.);
  std::uniform_real_distribution<double> uniform_dist_skull_compacta(4.33e-03, 6.86e-03);
  std::uniform_real_distribution<double> uniform_dist_skull_spongiosa(5.66e-03, 23.2e-03);

  
  // 
  // 
  simplex_.resize(4);
  for ( int i = 0 ; i < 4 ; i++ )
    {
      double 
	sigma_skin            = uniform_dist_skin(generator),
	sigma_skull_spongiosa = uniform_dist_skull_compacta(generator),
	sigma_skull_compact   = uniform_dist_skull_spongiosa(generator);
      
      simplex_[i] = std::make_tuple( 0.0, 
				     sigma_skin,
				     sigma_skull_spongiosa,
				     sigma_skull_compact,
				     false, false);
    }
}
//
//
//
void 
Solver::tCS_tDCS_local_conductivity::operator () ( )
{

  //////////////////////////////////////////////////////
  // Transcranial direct current stimulation equation //
  //////////////////////////////////////////////////////

  // 
  // incrementation loop
  int simplex_vertex = 0;
  int stop = 0;
  // 
  while( ++stop < 8 || !(std::get</* initialized */ 4 >(simplex_[0]) & 
			 std::get<4>(simplex_[1]) & std::get<4>(simplex_[2]) & 
			 std::get<4>(simplex_[3])) )
    {
      std::cout << "Yeah: " << simplex_vertex << " " << stop << std::endl;
      std::cout << (std::get</* initialized */ 4 >(simplex_[0]) & 
		    std::get<4>(simplex_[1]) & std::get<4>(simplex_[2]) & 
		    std::get<4>(simplex_[3])) << std::endl;
      for (int i = 0 ; i < simplex_.size()  ; i++ )
	std::cout << "~ " << i << " ~ " << std::get</* initialized */ 4 >(simplex_[i]) << std::endl;

      //
      // Update the conductivity
      if ( simplex_vertex < simplex_.size() )
	{
	  sigma_->conductivity_update( domains_, simplex_[simplex_vertex] );
	  std::get</* initialized */ 4 >(simplex_[simplex_vertex]) = true;
	  simplex_vertex++;
	}
      else
	{
	  int updated_simplex = -1;
	  for ( int i = 0 ; i < 4 ; i++ )
	    if ( std::get< /* updated */ 4 >(simplex_[i]) )
	      updated_simplex = i;
	  // 
	  if ( updated_simplex != -1)
	    sigma_->conductivity_update( domains_, simplex_[updated_simplex] );	  
	  else
	    {
	      std::cerr << "No update of the conductivity has been done" << std::endl;
	      abort();
	    }
	}

      //
      // tDCS electrical potential u
      //

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
      L.ds = *boundaries_;

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
      // Regulation terme:  \int u dx = 0
      double old_u_bar = 0.;
      double u_bar = 1.e+6;
      double U_bar = 0.;
      double N = u.vector()->size();
      int iteration = 0;
      double Sum = 1.e+6;
      //
      //  while ( abs( u_bar - old_u_bar ) > 0.1 )
      while ( fabs(Sum) > 1.e-3 )
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
 
      std::cout << "int u dx = " << Sum << std::endl;

 
      //
      // Filter function over the electrodes
      // solution_electrodes_extraction(u, electrodes_);
      electrodes_->get_current()->punctual_potential_evaluation(u, mesh_);

      std::cout << "electrode punctual CP6 " 
		<< electrodes_->get_current()->information( "CP6" ).get_V_() 
		<< std::endl;

      electrodes_->get_current()->surface_potential_evaluation(u, mesh_);


      std::cout << "electrode surface CP6 " 
		<< electrodes_->get_current()->information( "CP6" ).get_electrical_potential() 
		<< std::endl;

      // 
      // Estimate the the sum-of_squares
      // 
      if ( simplex_vertex > 3 )
	{}
      
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
    }
};
//
//
//
void
Solver::tCS_tDCS_local_conductivity::regulation_factor( const Function& u, 
							std::list<std::size_t>& Sub_domains)
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
