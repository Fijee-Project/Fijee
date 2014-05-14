#include <iostream>
#include "tCS_tDCS_local_conductivity.h"
inline double banana_helper(const double& x, const double& y) {
	return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}
double banana(Eigen::Vector3d& point) {
	int x = point[0];
	int y = point[1];
	int z = point[2];
	return banana_helper(x, y) + banana_helper(y, z);
}

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::tCS_tDCS_local_conductivity::tCS_tDCS_local_conductivity():
  Physics(), sample_(0)
{
  //
  // Define the function space
  V_.reset( new tCS_model::FunctionSpace(mesh_) );
  V_field_.reset( new tCS_field_model::FunctionSpace(mesh_) );

  // 
  // Output files
  // 
  
  // 
  // Head time series potential output file
  std::string file_head_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS_time_series.pvd");
  file_potential_time_series_ = nullptr; // new File( file_head_potential_ts_name.c_str() );
  // 
  std::string file_brain_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS_brain_time_series.pvd");
  file_brain_potential_time_series_ = nullptr; // new File( file_brain_potential_ts_name.c_str() );
  // 
  std::string file_filed_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS_field_time_series.pvd");
  file_field_time_series_ = nullptr; // new File( file_filed_potential_ts_name.c_str() );

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
      Eigen::Vector3d sigma;
      sigma << 
	uniform_dist_skin(generator),
	uniform_dist_skull_compacta(generator),
	uniform_dist_skull_spongiosa(generator);
      // 
      simplex_[i] = std::make_tuple( 0.0, sigma);
    }
   
  // 
  // Read measured potential
  std::cout << "Load electrical potential file" << std::endl;
  //
  std::string electrodes_xml = (SDEsp::get_instance())->get_files_path_measure_();
  electrodes_xml += "eeg_readout.xml";
  //
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file( electrodes_xml.c_str() );
  //
  switch( result.status )
    {
    case pugi::status_ok:
      {
	//
	// Check that we have a FIJEE XML file
	const pugi::xml_node fijee_node = xml_file.child("fijee");
	if (!fijee_node)
	  {
	    std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	    exit(1);
	  }

	// 
	// Get sampling
	const pugi::xml_node setup_node = fijee_node.child("setup");
	if (!setup_node)
	  {
	    std::cerr << "Read data from XML: no setup node" << std::endl;
	    exit(1);
	  }
	// Get the number of samples
	// loop over the samples
	for ( auto sample : setup_node )
	  {
	    //
	    // Get the number of electrodes
	    int sample_number  = sample.attribute("index").as_int();

	    //
	    //
	    for( auto electrode : sample )
	      {
		int index = electrode.attribute("index").as_uint();
		// Label
		std::string label = electrode.attribute("label").as_string(); 
		// Intensity
		double I = electrode.attribute("I").as_double(); /* Ampere */
		// Potential
		double V = electrode.attribute("V").as_double(); /* Volt */
		//
		electrodes_->get_current( sample_number )->add_measured_potential( label, V, I );
	      }
	  }

	//
	//
	break;
      };
    default:
      {
	std::cerr << "Error reading XML file: " << result.description() << std::endl;
	exit(1);
      }
    }
  
  // 
  // Minimizer 
  minimizer_algo_.reset( new Utils::Minimizers::Iterative_minimizer< Algorithm >());
}
//
//
//
void 
Solver::tCS_tDCS_local_conductivity::operator ()( )
{
//  //
//  // Mutex the electrodes vector poping process
//  //
//  int local_sample = 0;
//  try
//    {
//      // lock the electrode list
//      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
//      local_sample = sample_++;
//    }
//  catch (std::logic_error&)
//    {
//      std::cerr << "[exception caught]\n" << std::endl;
//    }
//
//  //
//  //
//  std::cout << "Sample " << local_sample << std::endl;
//  
//  //  //
//  //  // Define Dirichlet boundary conditions 
//  //  DirichletBC boundary_conditions(*V, source, perifery);

  // 
  // Initialize the simplex
  for( auto vertex = simplex_.begin() ; vertex != simplex_.end() ; vertex++ )
    {
      // 
      // Estimate the the sum-of_squares
      double tempo = solve( std::get<1>( *vertex ) );
      std::cout << tempo << std::endl; 
      std::get< 0 >(*vertex) = tempo;
    }

  
  for( auto vertex : simplex_ )
    std::cout << std::get<0>(vertex) << std::endl;

  	  
  // 
  // Minimize when estimations are done for 4 vertices of the tetrahedron simplex
  minimizer_algo_->initialization( std::bind( &tCS_tDCS_local_conductivity::solve, 
					      this, 
					      std::placeholders::_1 ), 
				   simplex_, conductivity_boundaries_ );
  minimizer_algo_->minimize();
}
double 
Solver::tCS_tDCS_local_conductivity::operator ()( const Eigen::Vector3d& A){return solve(A);};
//
//
//
double
Solver::tCS_tDCS_local_conductivity::solve( const Eigen::Vector3d& Vertex )
{
  // 
  // Update the conductivity
  sigma_->conductivity_update( domains_, Vertex );

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
  L.I  = *(electrodes_->get_current(0));
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
  // Mutex record potential at each electrods
  //
  try 
    {
      // lock the list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      //
      // Filter function over the electrodes
      // solution_electrodes_extraction(u, electrodes_);
      electrodes_->get_current(0)->punctual_potential_evaluation(u, mesh_);
	  
      // 
      // Estimate the the sum-of_squares
      return electrodes_->get_current(0)->sum_of_squares(); 
	  
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }
}
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
