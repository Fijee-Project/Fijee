#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <math.h>       /* round, floor, ceil, trunc */
#include <omp.h>
#include <cstdlib>
//
// UCSF
//
#include "Access_parameters.h"
#include "Build_mesh.h"
//
// Eigen
//
#include <Eigen/Dense>
//
//
//
// Voxel maximum distance from the centroid: (1.91)^2
#define VOXEL_LIMIT 3.65
//
// 600 000      BLOCKS  THREADS REMAIN
//   9 375       9375     64     0
//   4 687.5	 4687    128    64
//   2 343.75	 2343    256   192
//   1 171.875	 1171    512   448  
//
// Threads = 64
//#define THREADS   64
//#define BLOCKS  9375
//#define REMAIN     0
// Threads = 128
#define THREADS  128
#define BLOCKS  4687
#define REMAIN    64
// Threads = 256
//#define THREADS  256
//#define BLOCKS  2343
//#define REMAIN   192
//// Threads = 512
//#define THREADS  512
//#define BLOCKS  1171
//#define REMAIN   448
//
// We give a comprehensive type name
//
typedef Domains::Build_mesh Domains_build_mesh;
typedef Domains::Access_parameters DAp;
//
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;
//
// get function for the property map
//
Domains::Position_property_map::reference 
Domains::get( Domains::Position_property_map, Domains::Position_property_map::key_type p)
{
  return std::get<0>(p);
};

//
//
//
Domains_build_mesh::Build_mesh()
{}
//
//
//
void
Domains_build_mesh::Tetrahedrization()
{
  //
  // Loads image
  CGAL::Image_3 image;
  std::string head_model_inr = (Domains::Access_parameters::get_instance())->get_files_path_output_();
  head_model_inr += std::string("head_model.inr");
  image.read( head_model_inr.c_str() );

  //
  // Domain
  Mesh_domain domain(image);

  //
  // Mesh criteria
  // regular mesh
  //  Mesh_criteria criteria(facet_angle=30, facet_size=1.2, facet_distance=.8,
  //                         cell_radius_edge_ratio=2., cell_size=1.8);
  // Coarse in the midle fine on boundaries
//  Mesh_criteria criteria(facet_angle=25, facet_size=1., facet_distance=.5,
//                         cell_radius_edge_ratio=2., cell_size=8.);
  Mesh_criteria criteria(facet_angle=30, facet_size=1., facet_distance=.5,
                         cell_radius_edge_ratio=2., cell_size=8.);
  //  Mesh_criteria criteria(facet_angle=30, facet_size=2.5, facet_distance=1.5,
  //                         cell_radius_edge_ratio=2., cell_size=8.);
  
  //
  // Meshing
  mesh_ = CGAL::make_mesh_3<C3t3>(domain, criteria);
}
//
//
//
//Domains_build_mesh::Build_mesh( const Domains_build_mesh& that ):
//  pos_x_( that.get_pos_x() ),
//  pos_y_( that.get_pos_y() ),
//  tab_( new int[4] ),
//  list_position_ ( that.get_list_position() )
//{
//  std::copy( &that.get_tab(),  &that.get_tab() + 4, tab_ );
////  tab_[0] = ( &that.get_tab() )[0];
////  tab_[1] = ( &that.get_tab() )[1];
////  tab_[2] = ( &that.get_tab() )[2];
////  tab_[3] = ( &that.get_tab() )[3];
//}
//
//
//
//Domains_build_mesh::Build_mesh( Domains_build_mesh&& that ):
//  pos_x_( 0 ),
//  pos_y_( 0 ),
//  tab_( nullptr )
//{
//  // pilfer the source
//  list_position_ = std::move( that.list_position_ );
//  pos_x_ =  that.get_pos_x();
//  pos_y_ =  that.get_pos_y();
//  tab_   = &that.get_tab();
//  // reset that
//  that.set_pos_x( 0 );
//  that.set_pos_y( 0 );
//  that.set_tab( nullptr );
//}
//
//
//
Domains_build_mesh::~Build_mesh()
{
  /* Do nothing */
}
//
//
//
//Domains_build_mesh& 
//Domains_build_mesh::operator = ( const Domains_build_mesh& that )
//{
//  if ( this != &that ) 
//    {
//      // free existing ressources
//      if( tab_ )
//	{
//	  delete [] tab_;
//	  tab_ = nullptr;
//	}
//      // allocating new ressources
//      pos_x_ = that.get_pos_x();
//      pos_y_ = that.get_pos_y();
//      list_position_ = that.get_list_position();
//      //
//      tab_ = new int[4];
//      std::copy( &that.get_tab(),  &that.get_tab() + 4, tab_ );
//    }
//  //
//  return *this;
//}
//
//
//
void
Domains_build_mesh::operator ()( Mesh_output Output )
{

  switch ( Output ) 
    {
    case MESH_OUTPUT:
      {
	Output_mesh_format();
	break;
      }
    case MESH_SUBDOMAINS:
      {
	Output_FEniCS_xml();
	break;
      }
    case MESH_VTU:
      {
	Output_VTU_xml();
	break;
      }
    default:
      {
	std::cerr << "Mesh output unknown" << std::endl;
	abort();
      }
    }
}
//
//
//
void 
Domains_build_mesh::Output_FEniCS_xml()
{
  //
  // Transformation matrix
  //
  Eigen::Matrix< float, 3, 3 > rotation    = (DAp::get_instance())->get_rotation_();
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();

//  //
//  // typedef
//  typedef typename C3t3::Triangulation Triangulation;
//  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
//  typedef typename Triangulation::Vertex_handle Vertex_handle;
//  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
//  typedef typename Triangulation::Point Point_3;
//  //
//  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
//  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
//  // typedef typename No_patch_facet_pmap_second<C3t3,Cell_pmap> Facet_pmap_twice;
//  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;

  //
  // Output FEniCS xml files
  std::string output_mesh_XML              = (Domains::Access_parameters::get_instance())->get_files_path_output_();
  std::string output_subdomains_XML        = (Domains::Access_parameters::get_instance())->get_files_path_output_();
  std::string output_facets_subdomains_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_();
  //
  output_mesh_XML              += std::string("mesh.xml");
  output_subdomains_XML        += std::string("mesh_subdomains.xml");
  output_facets_subdomains_XML += std::string("mesh_facets_subdomains.xml");
  //
  std::ofstream FEniCS_xml_file(output_mesh_XML.c_str());
  std::ofstream FEniCS_xml_subdomains_file(output_subdomains_XML.c_str());
  std::ofstream FEniCS_xml_facets_subdomains_file(output_facets_subdomains_XML.c_str());
  FEniCS_xml_file << std::setprecision(20);
  
  //
  // header
  FEniCS_xml_file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  FEniCS_xml_file << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
  FEniCS_xml_file << "  <mesh celltype=\"tetrahedron\" dim=\"3\">" << std::endl;
  //
  FEniCS_xml_subdomains_file << "<?xml version=\"1.0\"?>" << std::endl;
  FEniCS_xml_subdomains_file << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
  FEniCS_xml_subdomains_file << "  <mesh_function>" << std::endl;
  //
  FEniCS_xml_facets_subdomains_file << "<?xml version=\"1.0\"?>" << std::endl;
  FEniCS_xml_facets_subdomains_file << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
//  FEniCS_xml_facets_subdomains_file << "  <mesh_function>" << std::endl;

  // FEniCS_xml_file << "" << std::endl;

  //
  // Vertices
  const Triangulation& triangulation = mesh_.triangulation();
  Cell_pmap            cell_pmap( mesh_ );
  Facet_pmap           facet_pmap( mesh_, cell_pmap );
  Vertex_pmap          vertex_pmap( mesh_, cell_pmap, facet_pmap );
  //
  FEniCS_xml_file << "    <vertices size=\"" << triangulation.number_of_vertices() << "\">" << std::endl;


  //
  //
  std::map<Vertex_handle, int> V;
  int inum = 0;
  for( Finite_vertices_iterator vit = triangulation.finite_vertices_begin();
       vit != triangulation.finite_vertices_end();
       ++vit)
    {
      Eigen::Matrix< float, 3, 1 > position;
      position <<
	(float)vit->point().x(),
	(float)vit->point().y(),
	(float)vit->point().z();
      //
      position = rotation * position + translation;
      // [mm] to [m]
      position *= 1.e-3;
      //      Point_3 p = vit->point();
      FEniCS_xml_file << "      <vertex index=\"" << inum++ << "\" x=\""
		      << position(0,0) << "\" y=\""
		      << position(1,0) << "\" z=\""
		      << position(2,0) << "\"/>"
	// << vertex_pmap.index( vit ) << " "
		      << std::endl;
      //
      V[vit] = inum;
    }


//  //
//  //
//  std::map<int, C3t3::Facet> F;
//  inum = 0;
//  for( Facet_iterator fit = mesh_.facets_in_complex_begin() ;
//       fit != mesh_.facets_in_complex_end() ;
//       ++fit )
//    F[inum++] = *fit;


  // 
  // End of vertices
  FEniCS_xml_file << "    </vertices>" << std::endl;

  //
  // Tetrahedra
  FEniCS_xml_file << "    <cells size=\"" << mesh_.number_of_cells_in_complex() << "\">" << std::endl;
  //
  FEniCS_xml_subdomains_file << "    <mesh_value_collection type=\"uint\" dim=\"3\" size=\"" 
			     << mesh_.number_of_cells_in_complex() << "\">" << std::endl;
  //
  FEniCS_xml_facets_subdomains_file << "    <mesh_value_collection  name=\"f\" type=\"uint\" dim=\"2\" size=\"" 
				    << 4 * mesh_.number_of_cells_in_complex() << "\">" << std::endl;
  //
  int ifacet = 0;
  inum = 0;
  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ;
       cit != mesh_.cells_in_complex_end() ;
       ++cit )
    {
      FEniCS_xml_file << "      <tetrahedron index=\"" << inum << "\" v0=\""
		      << V[cit->vertex( 0 )] - 1 << "\" v1=\""
		      << V[cit->vertex( 1 )] - 1 << "\" v2=\""
		      << V[cit->vertex( 2 )] - 1 << "\" v3=\""
		      << V[cit->vertex( 3 )] - 1 << "\"/>"
		      << std::endl;
      //
      FEniCS_xml_subdomains_file << "      <value cell_index=\"" << inum 
				 << "\" local_entity=\"0\" value=\"" 
				 << cell_pmap.subdomain_index( cit ) << "\" />"
				 << std::endl;
      //
      for (int vertex = 0 ; vertex < 4 ; vertex++ )
	{
	  FEniCS_xml_facets_subdomains_file << "      <value cell_index=\"" << inum
					    << "\" local_entity=\"" << ifacet++
					    << "\" value=\"";
	  //
	  FEniCS_xml_facets_subdomains_file << cell_pmap.subdomain_index( cit )  << "\" />"
//	  FEniCS_xml_facets_subdomains_file << facet_pmap.surface_index( C3t3::Facet(cit, vertex) ) << "\" />"
					    << std::endl;
   
	}
      //
      ifacet = 0;
      inum++;
    }


//  //
//  // Facets
//  FEniCS_xml_facets_subdomains_file << "    <mesh_value_collection type=\"uint\" dim=\"3\" size=\"" 
//				    << mesh_.number_of_facets_in_complex() << "\">" << std::endl;
//  //
//  inum = 0;
//  int ifacet = 0;
//  for( Facet_iterator fit = mesh_.facets_in_complex_begin() ;
//       fit != mesh_.facets_in_complex_end() ;
//       ++fit )
//    {
//      //
//      FEniCS_xml_facets_subdomains_file << "      <value cell_index=\"" << inum 
//					<< "\" local_entity=\"" << ifacet++
//					<< "\" value=\"" 
//					<< facet_pmap.surface_index( *fit ) << "\" />"
//					<< std::endl;
//      //
//      if (ifacet == 4) 
//	{
//	  ifacet = 0;
//	  inum++;
//	}
//    }

  //
  // End of tetrahedra
  FEniCS_xml_file << "    </cells>" << std::endl;
  //
  FEniCS_xml_subdomains_file << "    </mesh_value_collection>" << std::endl;
  //
  FEniCS_xml_facets_subdomains_file << "    </mesh_value_collection>" << std::endl;

  //
  // Tail
  FEniCS_xml_file << "  </mesh>" << std::endl;
  FEniCS_xml_file << "</dolfin>" << std::endl;
  //
  FEniCS_xml_subdomains_file << "  </mesh_function>" << std::endl;
  FEniCS_xml_subdomains_file << "</dolfin>" << std::endl;
  //
//  FEniCS_xml_facets_subdomains_file << "  </mesh_function>" << std::endl;
  FEniCS_xml_facets_subdomains_file << "</dolfin>" << std::endl;

  //  //
  //  //
  //  FEniCS_xml_file.close();
  //  FEniCS_xml_subdomains_file.close();
  //  FEniCS_xml_facets_subdomains_file.close();
}
//
//
//
void 
Domains_build_mesh::Output_VTU_xml()
{
  //
  // Transformation matrix
  //
  Eigen::Matrix< float, 3, 3 > rotation = (DAp::get_instance())->get_rotation_();
  //
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();

//  //
//  // typedef
//  typedef typename C3t3::Triangulation Triangulation;
//  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
//  typedef typename Triangulation::Vertex_handle Vertex_handle;
//  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
//  typedef typename Triangulation::Point Point_3;
//  //
//  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
//  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
//  // typedef typename No_patch_facet_pmap_second<C3t3,Cell_pmap> Facet_pmap_twice;
//  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;

  //
  // Output FEniCS xml file
  std::ofstream VTU_xml_file("mesh.vtu");
  
  //
  // Vertices
  Cell_pmap            cell_pmap( mesh_ );
  Facet_pmap           facet_pmap( mesh_, cell_pmap );
  Vertex_pmap          vertex_pmap( mesh_, cell_pmap, facet_pmap );
  //
  std::map<Vertex_handle, int> V;
  int 
    inum = 0,
    num_tetrahedra = 0;
  int offset = 0;
  std::string
    point_data,
    vertices_position_string,
    vertices_associated_to_tetra_string,
    offsets_string,
    cells_type_string;
  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ;
       cit != mesh_.cells_in_complex_end() ;
       ++cit )
    {
      //
      // Vertex position
      Eigen::Matrix< float, 3, 1 > position_0;
      position_0 <<
	(float)cit->vertex( 0 )->point().x(),
	(float)cit->vertex( 0 )->point().y(),
	(float)cit->vertex( 0 )->point().z();
      //
      Eigen::Matrix< float, 3, 1 > position_1;
      position_1 <<
	(float)cit->vertex( 1 )->point().x(),
	(float)cit->vertex( 1 )->point().y(),
	(float)cit->vertex( 1 )->point().z();
      //
      Eigen::Matrix< float, 3, 1 > position_2;
      position_2 <<
	(float)cit->vertex( 2 )->point().x(),
	(float)cit->vertex( 2 )->point().y(),
	(float)cit->vertex( 2 )->point().z();
      //
      Eigen::Matrix< float, 3, 1 > position_3;
      position_3 <<
	(float)cit->vertex( 3 )->point().x(),
	(float)cit->vertex( 3 )->point().y(),
	(float)cit->vertex( 3 )->point().z();
      //
      position_0 = rotation * position_0 + translation;
      position_1 = rotation * position_1 + translation;
      position_2 = rotation * position_2 + translation;
      position_3 = rotation * position_3 + translation;
      
      //
      //
      if ( position_0(2,0) > 35. && position_0(2,0) < 50. &&
	   position_1(2,0) > 35. && position_1(2,0) < 50. &&
	   position_2(2,0) > 35. && position_2(2,0) < 50. &&
	   position_3(2,0) > 35. && position_3(2,0) < 50.  )
	{
	  
	  //
	  //  vertex id
	  if( V[ cit->vertex( 0 ) ] == 0 )
	    {
	      V[ cit->vertex( 0 ) ] = ++inum;
	      vertices_position_string += 
		std::to_string( position_0(0,0) ) + " " + 
		std::to_string( position_0(1,0) ) + " " +
		std::to_string( position_0(2,0) ) + " " ;
	      point_data += std::to_string( vertex_pmap.index( cit->vertex( 0 ) ) / 19. ) + " ";
	    }
	  if( V[ cit->vertex( 1 ) ] == 0 )
	    {
	      V[ cit->vertex( 1 ) ] = ++inum;
	      vertices_position_string += 
		std::to_string( position_1(0,0) ) + " " + 
		std::to_string( position_1(1,0) ) + " " +
		std::to_string( position_1(2,0) ) + " " ;
	      point_data += std::to_string( vertex_pmap.index( cit->vertex( 1 ) ) / 19. ) + " ";
	    }
	  if( V[ cit->vertex( 2 ) ] == 0 )
	    {
	      V[ cit->vertex( 2 ) ] = ++inum;
	      vertices_position_string += 
		std::to_string( position_2(0,0) ) + " " + 
		std::to_string( position_2(1,0) ) + " " +
		std::to_string( position_2(2,0) ) + " " ;
	      point_data += std::to_string( vertex_pmap.index( cit->vertex( 2 ) ) / 19. ) + " ";
	    }
	  if( V[ cit->vertex( 3 ) ] == 0 )
	    {
	      V[ cit->vertex( 3 ) ] = ++inum;
	      vertices_position_string += 
		std::to_string( position_3(0,0) ) + " " + 
		std::to_string( position_3(1,0) ) + " " +
		std::to_string( position_3(2,0) ) + " " ;
	      point_data += std::to_string( vertex_pmap.index( cit->vertex( 3 ) ) / 19. ) + " ";
	    }
	  
	  //
	  // Volume associated
	  vertices_associated_to_tetra_string += 
	    std::to_string( V[cit->vertex( 0 )] - 1 ) + " " +
	    std::to_string( V[cit->vertex( 1 )] - 1 ) + " " +
	    std::to_string( V[cit->vertex( 2 )] - 1 ) + " " +
	    std::to_string( V[cit->vertex( 3 )] - 1 ) + " ";
	  
	  //
	  // Offset for each volumes
	  offset += 4;
	  offsets_string += std::to_string( offset ) + " ";
	  //
	  cells_type_string += "10 ";
	  //
	  num_tetrahedra++;
	}
    }

  //
  // header
  VTU_xml_file << "<?xml version=\"1.0\"?>" << std::endl;
  VTU_xml_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
  VTU_xml_file << "  <UnstructuredGrid>" << std::endl;

  // 
  // End of vertices
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

  //
  //
  std::cout << "num vertices: " << inum << std::endl;
  std::cout << "num volumes: " << num_tetrahedra << std::endl;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const Domains_build_mesh& that)
{
  //  std::for_each( that.get_list_position().begin(),
  //		 that.get_list_position().end(),
  //		 [&stream]( int Val )
  //		 {
  //		   stream << "list pos = " << Val << "\n";
  //		 });
  //  //
  //  stream << "position x = " <<    that.get_pos_x() << "\n";
  //  stream << "position y = " <<    that.get_pos_y() << "\n";
  //  if ( &that.get_tab() )
  //    {
  //      stream << "tab[0] = "     << ( &that.get_tab() )[0] << "\n";
  //      stream << "tab[1] = "     << ( &that.get_tab() )[1] << "\n";
  //      stream << "tab[2] = "     << ( &that.get_tab() )[2] << "\n";
  //      stream << "tab[3] = "     << ( &that.get_tab() )[3] << "\n";
  //    }
  //
  return stream;
};
