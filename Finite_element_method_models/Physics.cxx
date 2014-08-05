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
#include "Physics.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::Physics::Physics()
{
  //
  // Load the mesh
  std::cout << "Load mesh file" << std::endl;
  //
  std::string mesh_xml = (SDEsp::get_instance())->get_files_path_output_();
  mesh_xml += "mesh.xml";
  //
  mesh_.reset( new dolfin::Mesh(mesh_xml) );
  //
  info( *mesh_ );

  //
  // Load Sub_domains
  std::cout << "Load Sub_domains file" << std::endl;
  //
  std::string subdomains_xml = (SDEsp::get_instance())->get_files_path_output_();
  subdomains_xml += "mesh_subdomains.xml";
  //
  domains_.reset( new MeshFunction< long unsigned int >(mesh_, subdomains_xml) );
  // write domains
  std::string domains_file_name = (SDEsp::get_instance())->get_files_path_result_();
  domains_file_name            += std::string("domains.pvd");
  File domains_file( domains_file_name.c_str() );
  domains_file << *domains_;

  //
  // Load the conductivity. Anisotrope conductivity
  std::cout << "Load conductivity files" << std::endl;
  sigma_.reset( new Solver::Tensor_conductivity(mesh_) );

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
}
//
//
//
void
Solver::Physics::solution_domain_extraction( const dolfin::Function& u, 
					     std::list<std::size_t>& Sub_domains,
					     const char* name)
{
  // 
  const std::size_t num_vertices = mesh_->num_vertices();
  
  // Get number of components
  const std::size_t dim = u.value_size();
  const std::size_t rank = u.value_rank();

  // Open file
  std::string sub_dom(name);
  for ( auto sub : Sub_domains )
    sub_dom += std::string("_") + std::to_string(sub);
  sub_dom += std::string(".vtu");
  //
  std::string extracted_solution = (SDEsp::get_instance())->get_files_path_result_();
  extracted_solution            += sub_dom;
  //
  std::ofstream VTU_xml_file(extracted_solution.c_str(), std::ofstream::out);
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
  for ( dolfin::CellIterator cell(*mesh_) ; !cell.end() ; ++cell )
    // loop over extraction sub-domains
    for( auto sub_domain : Sub_domains ) 
      if ( (*domains_)[cell->index()] == sub_domain )
	{
	  //  vertex id
	  for ( dolfin::VertexIterator vertex(*cell) ; !vertex.end() ; ++vertex )
	    {
	      if( V[ vertex->index() ] == -1 )
		{
		  //
		  V[ vertex->index() ] = inum++;
		  vertices_position_string += 
		    std::to_string( vertex->point().x() ) + " " + 
		    std::to_string( vertex->point().y() ) + " " +
		    std::to_string( vertex->point().z() ) + " " ;

		  //
		  int index = vertex->index();
		  for ( int i = 0 ; i < dim ; i ++)
		    point_data += std::to_string( values[index + i*num_vertices] ) + " ";
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
  if (rank == 0) 
  {
    VTU_xml_file << "<PointData  Scalars=\"" << u.name() << "\"> " << std::endl;
    VTU_xml_file << "<DataArray  type=\"Float32\"  Name=\"" << u.name() 
		 << "\"  format=\"ascii\">" << std::endl;
  }
  else if (rank == 1)
  {
    VTU_xml_file << "<PointData  Vectors=\"" << u.name() << "\"> " << std::endl;
    VTU_xml_file << "<DataArray  type=\"Float32\"  Name=\"" << u.name() 
		 << "\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  }
  else if (rank == 2)
  {
    VTU_xml_file << "<PointData  Tensors=\"" << u.name() << "\"> " << std::endl;
    VTU_xml_file << "<DataArray  type=\"Float32\"  Name=\"" << u.name() 
		 << "\"  NumberOfComponents=\"9\" format=\"ascii\">" << std::endl;
  }
  //
  VTU_xml_file << point_data;
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
