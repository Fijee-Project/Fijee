#include <iostream>
#include "tCS_tDCS.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::tCS_tDCS::tCS_tDCS():
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
  file_potential_time_series_ = new File( file_head_potential_ts_name.c_str() );
  // 
  std::string file_brain_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS_brain_time_series.pvd");
   file_brain_potential_time_series_ = nullptr; // new File( file_brain_potential_ts_name.c_str() );
  // 
  std::string file_filed_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS_field_time_series.pvd");
   file_field_time_series_ = nullptr; // new File( file_filed_potential_ts_name.c_str() );
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


  //////////////////////////////////////////////////////
  // Transcranial direct current stimulation equation //
  //////////////////////////////////////////////////////
      

  //
  // tDCS electric potential u
  //

  //  //
  //  // PDE boundary conditions
  //  DirichletBC bc(*V_, *(electrodes_->get_current(0)), *boundaries_, 101);

  //
  // Define variational forms
  tCS_model::BilinearForm a(V_, V_);
  tCS_model::LinearForm L(V_);
      
  //
  // Anisotropy
  // Bilinear
  a.a_sigma  = *sigma_;
  // a.dx       = *domains_;
  //  std::cout << electrodes_->get_current(0)->information("T7").get_I_() << std::endl;  
  
  // Linear
  L.I  = *(electrodes_->get_current( /*local_sample*/ 0));
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
  while ( fabs(Sum) > 1.e-6 )
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
  // 
  std::cout << "int u dx = " << Sum << std::endl;
 
  //
  // Filter function over a subdomain
  std::list<std::size_t> test_sub_domains{4,5};
  solution_domain_extraction(u, test_sub_domains, "tDCS_potential");
//  //
//  // Filter function over a subdomain
//  std::list<std::size_t> test_sub_domains{4,5};
//  solution_domain_extraction(u, test_sub_domains, file_brain_potential_time_series_);

  //
  // Mutex record potential at each electrods
  //
  try 
    {
      // lock the dipole list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      // 
      // electrodes_->get_current(0)->punctual_potential_evaluation(u, mesh_);
      electrodes_->get_current(0)->surface_potential_evaluation(u, mesh_);
      electrodes_->record_potential( /*dipole idx*/ 0, 
				     /*time   idx*/ 0);
    }
  catch (std::logic_error&) 
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }


  //
  // Mutex in the critical zone
  //
  try
    {
      // lock the electrode list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      *file_potential_time_series_ << u;
    }
  catch (std::logic_error&)
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }

  //
  // tDCS electric current density field \vec{J}
  // 
  
  if (true)
    {
      //
      // Define variational forms
      tCS_field_model::BilinearForm a_field(V_field_, V_field_);
      tCS_field_model::LinearForm L_field(V_field_);
      
      //
      // Anisotropy
      // Bilinear
      // a.dx       = *domains_;
      
      
      // Linear
      L_field.u       = u;
      L_field.a_sigma = *sigma_;
      //  L.ds = *boundaries_;
      
      //
      // Compute solution
      Function J(*V_field_);
      LinearVariationalProblem problem_field(a_field, L_field, J/*, bc*/);
      LinearVariationalSolver  solver_field(problem_field);
      // krylov
      solver_field.parameters["linear_solver"]  
	= (SDEsp::get_instance())->get_linear_solver_();
      solver_field.parameters("krylov_solver")["maximum_iterations"] 
	= (SDEsp::get_instance())->get_maximum_iterations_();
      solver_field.parameters("krylov_solver")["relative_tolerance"] 
	= (SDEsp::get_instance())->get_relative_tolerance_();
      solver_field.parameters["preconditioner"] 
	= (SDEsp::get_instance())->get_preconditioner_();
      //
      solver_field.solve();
      
      // 
      // Output
      solution_domain_extraction(J, test_sub_domains, "tDCS_Current_density");
      //
      std::string field_name = (SDEsp::get_instance())->get_files_path_result_() + 
	std::string("tDCS_field.pvd");
      File field( field_name.c_str() );
      //
      field << J;
    }

 
 //
 // Validation process
 // !! WARNING !!
 // This process must be pluged only for spheres model
 if( false )
   {
     std::cout << "Validation processing" << std::endl;
     //
     // TODO CHANGE THE PROTOTYPE
     std::cout << "je passe create T7" << std::endl;
//     Solver::Spheres_electric_monopole mono_T7( (*electrodes_)["T7"].get_I_(),  
//						(*electrodes_)["T7"].get_r0_projection_() );
     Solver::Spheres_electric_monopole mono_T7( electrodes_->get_current(0)->information("T7").get_I_(),  
						electrodes_->get_current(0)->information("T7").get_r0_projection_() );
     //
     std::cout << "je passe create T8" << std::endl;
//     Solver::Spheres_electric_monopole mono_T8( (*electrodes_)["T8"].get_I_(),  
//						(*electrodes_)["T8"].get_r0_projection_() );
     Solver::Spheres_electric_monopole mono_T8( electrodes_->get_current(0)->information("T8").get_I_(),  
						electrodes_->get_current(0)->information("T8").get_r0_projection_() );
     //
     //
     Function 
       Injection_T7(V_),
       Injection_T8(V_);
     //Function & Injection = Injection_T7;

     //
     std::cout << "je passe T7" << std::endl;
     Injection_T7.interpolate( mono_T7 );
     // std::thread T7(Injection_T7.interpolate, mono_T7);
     std::cout << "je passe T8" << std::endl;
     Injection_T8.interpolate( mono_T8 );
     //
     std::cout << "je passe T7+T8" << std::endl;
     *(Injection_T7.vector()) += *(Injection_T8.vector());
     std::cout << "je passe T7+T8 end" << std::endl;

     //
     // Produce outputs
     std::string file_validation = (SDEsp::get_instance())->get_files_path_result_() + 
       std::string("validation.pvd");
     File validation( file_validation.c_str() );
     //
     validation << Injection_T7;
     std::cout << "je passe" << std::endl;
   }
};
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
