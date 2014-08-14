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
#include <iostream>
#include "tCS_tACS.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::tCS_tACS::tCS_tACS():
  Physics(), sample_(0)
{
  //
  // Define the function space
  V_.reset( new tCS_model::FunctionSpace(mesh_) );
  V_current_density_.reset( new tCS_current_density_model::FunctionSpace(mesh_) );
  V_E_.reset( new tCS_electrical_field_model::FunctionSpace(mesh_) );


  // 
  // Output files
  // 
  
  // 
  // Head time series potential output file
  std::string file_head_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tACS_time_series.pvd");
  file_potential_time_series_.reset( new File( file_head_potential_ts_name.c_str() ) );
  // 
  std::string file_brain_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tACS_brain_time_series.pvd");
  file_brain_potential_time_series_.reset( new File( file_brain_potential_ts_name.c_str() ) );
  // 
  std::string file_filed_potential_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tACS_current_density_time_series.pvd");
  file_current_density_time_series_.reset( new File( file_filed_potential_ts_name.c_str() ) );
  // 
  std::string file_E_ts_name = (SDEsp::get_instance())->get_files_path_result_() + 
    std::string("tDCS_E_time_series.pvd");
  file_E_time_series_.reset(new File(file_E_ts_name.c_str()));
  
  // 
  // Parcellation information
  // 
  tCS_E_parcel_.reset( new Parcellation_information< Field< 3 > >("Electric_field") );
}
//
//
//
void 
Solver::tCS_tACS::operator () ( /*Solver::Phi& source,
				  SLD_model::FunctionSpace& V,
				  FacetFunction< size_t >& boundaries*/)
{
  //
  // Mutex the electrodes vector poping process
  //
  int local_sample = 0;
  try
    {
      // lock the electrode list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      local_sample = sample_++;
    }
  catch (std::logic_error&)
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }

  //
  //
  std::cout << "Sample " << local_sample << std::endl;
  
  //  //
  //  // Define Dirichlet boundary conditions 
  //  DirichletBC boundary_conditions(*V, source, perifery);


  //////////////////////////////////////////////////////
  // Transcranial direct current stimulation equation //
  //////////////////////////////////////////////////////
      

  //
  // tACS electric potential u
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
  
  
  // Linear
  L.I  = *( electrodes_->get_current(local_sample) );
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
  solution_domain_extraction(u, test_sub_domains, file_brain_potential_time_series_);

  //
  // tACS electric potential
  //
  try
    {
      // lock the electrode list
      std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
      // 
      // electrodes_->get_current(local_sample)->punctual_potential_evaluation(u, mesh_);
      std::cout << "time av record: " << electrodes_->get_current(local_sample)->get_time_() << std::endl;
      electrodes_->get_current(local_sample)->surface_potential_evaluation(u, mesh_);
      electrodes_->record_potential( /*dipole idx*/ 0,
				     electrodes_->get_current(local_sample)->get_time_() );
      std::cout << "time ap record: " << electrodes_->get_current(local_sample)->get_time_() << std::endl;
    }
  catch (std::logic_error&)
    {
      std::cerr << "[exception caught]\n" << std::endl;
    }
  
  // output
  if(false)
    {
      try
	{
	  // lock the electrode list
	  std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	  *file_potential_time_series_ 
	    << std::make_pair<const Function*, double>(&u, 
						       electrodes_->get_current(local_sample)->get_time_() );
	}
      catch (std::logic_error&)
	{
	  std::cerr << "[exception caught]\n" << std::endl;
	}
    }
 
  // 
  // tACS electric current density field \vec{J}
  // 

  if (false)
    {
      //
      // Define variational forms
      tCS_current_density_model::BilinearForm a_field(V_current_density_, V_current_density_);
      tCS_current_density_model::LinearForm L_field(V_current_density_);
      
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
      Function J(*V_current_density_);
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
      // Filter function over a subdomain
      //
      // Mutex in the critical zone
      //
      try
	{
	  // lock the electrode list
	  std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	  *file_current_density_time_series_ 
	    << std::make_pair<const Function*, double>(&J, 
						       electrodes_
						       ->get_current(local_sample)
						       ->get_time_() );
	}
      catch (std::logic_error&)
	{
	  std::cerr << "[exception caught]\n" << std::endl;
	}
    }

 
  //
  // tACS electrical field \vec{E}
  // 
  
  if (true)
    {
      //
      // Define variational forms
      tCS_electrical_field_model::BilinearForm a_E(V_E_, V_E_);
      tCS_electrical_field_model::LinearForm L_E(V_E_);
      
      //
      // Anisotropy
      // Bilinear
      // a.dx       = *domains_;
      // a.ds = *boundaries_;
      
      
      // Linear
      L_E.u  = u;
      //      L_Er.a_sigma = *sigma_;
      //       L.ds = *boundaries_;
      
      //
      // Compute solution
      Function E(*V_E_);
      LinearVariationalProblem problem_E(a_E, L_E, E/*, bc*/);
      LinearVariationalSolver  solver_E(problem_E);
      // krylov
      solver_E.parameters["linear_solver"]  
	= (SDEsp::get_instance())->get_linear_solver_();
      solver_E.parameters("krylov_solver")["maximum_iterations"] 
	= (SDEsp::get_instance())->get_maximum_iterations_();
      solver_E.parameters("krylov_solver")["relative_tolerance"] 
	= (SDEsp::get_instance())->get_relative_tolerance_();
      solver_E.parameters["preconditioner"] 
	= (SDEsp::get_instance())->get_preconditioner_();
      //
      solver_E.solve();
      

      // 
      // Output
      //
      //
      // Mutex in the critical zone
      //
      try
	{
	  // lock the electrode list
	  std::lock_guard< std::mutex > lock_critical_zone ( critical_zone_ );
	  // 
	  // Evaluation
	  Array<double> position(3);
	  Array<double> val_1(3);
	  Array<double> val_2(3);
	  position[0] = -0.0450098;
	  position[1] =  0.0170047;
	  position[2] =  0.0353029;
	  int cell = 1740037;
	  E.eval(val_1, position);
	  //       E.eval(val_2, position, cell)
	  
	  std::cout << "val at (" << -0.0450098 << "," << 0.0170047 << "," << 0.0353029 << ") = " 
		    << val_1[0] << " " 
		    << val_1[1] << " " 
		    << val_1[2] << " " 
		    << std::endl;


	  *file_E_time_series_  
	    << std::make_pair<const Function*, double>(&E, 
						       electrodes_
						       ->get_current(local_sample)
						       ->get_time_() );
	}
      catch (std::logic_error&)
	{
	  std::cerr << "[exception caught]\n" << std::endl;
	}
    }
};
//
//
//
void
Solver::tCS_tACS::regulation_factor(const Function& u, std::list<std::size_t>& Sub_domains)
{
  // 
  const std::size_t num_vertices = mesh_->num_vertices();
  
  // Get number of components
  const std::size_t dim = u.value_size();
  
  // Open file
  std::string sub_dom("tACS");
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
//
// 
//
void 
Solver::tCS_tACS::solution_domain_extraction( const dolfin::Function& U, 
					      const std::list<std::size_t>& Sub_domains,
					      std::shared_ptr<File> File )
{
  // TODO Build a new time series to extract domains
  // Don't forget to call the mutex while writting the output file.
}
