#include "Physics.h"

typedef Solver::PDE_solver_parameters SDEsp;

//
//
//
Solver::Physics::Physics()
{
  //
  // Load the mesh
  std::cout << "Load the mesh" << std::endl;
  //
  std::string mesh_xml = (SDEsp::get_instance())->get_files_path_output_();
  mesh_xml += "mesh.xml";
  //
  mesh_.reset( new dolfin::Mesh(mesh_xml.c_str()) );
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
