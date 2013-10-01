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
{
  //
  // Loads image
  CGAL::Image_3 image;
  image.read("head_model.inr");

  //
  // Domain
  Mesh_domain domain(image);

  //
  // Mesh criteria
  // regular mesh
  //  Mesh_criteria criteria(facet_angle=30, facet_size=1.2, facet_distance=.8,
  //                         cell_radius_edge_ratio=2., cell_size=1.8);
  // Coarse in the midle fine on boundaries
  Mesh_criteria criteria(facet_angle=25, facet_size=1., facet_distance=.5,
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
//Domains_build_mesh& 
//Domains_build_mesh::operator = ( Domains_build_mesh&& that )
//{
//  if( this != &that )
//    {
//      // initialisation
//      pos_x_ = 0;
//      pos_y_ = 0;
//      delete [] tab_;
//      tab_   = nullptr;
//      // pilfer the source
//      list_position_ = std::move( that.list_position_ );
//      pos_x_ =  that.get_pos_x();
//      pos_y_ =  that.get_pos_y();
//      tab_   = &that.get_tab();
//      // reset that
//      that.set_pos_x( 0 );
//      that.set_pos_y( 0 );
//      that.set_tab( nullptr );
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
    case MESH_CONDUCTIVITY:
      {
	Output_mesh_conductivity_xml();
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
  std::ofstream FEniCS_xml_file("mesh.xml");
  std::ofstream FEniCS_xml_subdomains_file("mesh_subdomains.xml");
  FEniCS_xml_file << std::setprecision(20);
  
  //
  // header
  FEniCS_xml_file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl << std::endl;
  FEniCS_xml_file << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
  FEniCS_xml_file << "  <mesh celltype=\"tetrahedron\" dim=\"3\">" << std::endl;
  //
  FEniCS_xml_subdomains_file << "<?xml version=\"1.0\"?>" << std::endl << std::endl;
  FEniCS_xml_subdomains_file << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
  FEniCS_xml_subdomains_file << "  <mesh_function>" << std::endl;

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
      FEniCS_xml_subdomains_file << "      <value cell_index=\"" << inum++ 
				 << "\" local_entity=\"0\" value=\"" 
				 << cell_pmap.subdomain_index( cit ) << "\" />"
				 << std::endl;
    }

  //
  // End of tetrahedra
  FEniCS_xml_file << "    </cells>" << std::endl;
  //
  FEniCS_xml_subdomains_file << "    </mesh_value_collection>" << std::endl;

  //
  // Tail
  FEniCS_xml_file << "  </mesh>" << std::endl;
  FEniCS_xml_file << "</dolfin>" << std::endl;
  //
  FEniCS_xml_subdomains_file << "  </mesh_function>" << std::endl;
  FEniCS_xml_subdomains_file << "</dolfin>" << std::endl;

  //  //
  //  //
  //  FEniCS_xml_file.close();
  //  FEniCS_xml_subdomains_file.close();
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
void 
Domains_build_mesh::Output_mesh_conductivity_xml()
{
  //
  // Output FEniCS conductivity xml files. 
  // We fillup the triangular sup from the symetric conductivity tensor
  std::ofstream 
    FEniCS_xml_C00("C00.xml"), FEniCS_xml_C01("C01.xml"), FEniCS_xml_C02("C02.xml"), 
    FEniCS_xml_C11("C11.xml"), FEniCS_xml_C12("C12.xml"), 
    FEniCS_xml_C22("C22.xml");
  //
  int num_of_tetrahedra = list_coefficients_.size();
  

  //
  // header
  FEniCS_xml_C00 
    << "<?xml version=\"1.0\"?> \n <dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">\n  "
    << "<mesh_function>\n    <mesh_value_collection type=\"double\" dim=\"3\" size=\"" 
    << num_of_tetrahedra << "\">\n";
  //
  FEniCS_xml_C01 
    << "<?xml version=\"1.0\"?> \n<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">\n  "
    << "<mesh_function>\n    <mesh_value_collection type=\"double\" dim=\"3\" size=\"" 
    << num_of_tetrahedra << "\">\n";
  //
  FEniCS_xml_C02 
    << "<?xml version=\"1.0\"?> \n<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">\n  "
    << "<mesh_function>\n    <mesh_value_collection type=\"double\" dim=\"3\" size=\"" 
    << num_of_tetrahedra << "\">\n";
  //
  FEniCS_xml_C11 
    << "<?xml version=\"1.0\"?> \n<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">\n  "
    << "<mesh_function>\n    <mesh_value_collection type=\"double\" dim=\"3\" size=\"" 
    << num_of_tetrahedra << "\">\n";
  //
  FEniCS_xml_C12 
    << "<?xml version=\"1.0\"?> \n<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">\n  "
    << "<mesh_function>\n    <mesh_value_collection type=\"double\" dim=\"3\" size=\"" 
    << num_of_tetrahedra << "\">\n";
  //
  FEniCS_xml_C22 
    << "<?xml version=\"1.0\"?> \n<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">\n  "
    << "<mesh_function>\n    <mesh_value_collection type=\"double\" dim=\"3\" size=\"" 
    << num_of_tetrahedra << "\">\n";

  
  //
  // Main loop
  for ( auto it = list_coefficients_.begin() ; 
	it != list_coefficients_.end() ; 
	++it )
    {
      //
      // C00
      FEniCS_xml_C00 
	<< "      <value cell_index=\"" <<  it->cell_id
	<< "\" local_entity=\"0\" value=\""
	<< it->conductivity_coefficients[0]
	<< "\" />\n";
 
      //
      // C01
      FEniCS_xml_C01 
	<< "      <value cell_index=\"" <<  it->cell_id
	<< "\" local_entity=\"0\" value=\""
	<< it->conductivity_coefficients[1]
	<< "\" />\n";
 
      //
      // C02
      FEniCS_xml_C02 
	<< "      <value cell_index=\"" <<  it->cell_id
	<< "\" local_entity=\"0\" value=\""
	<< it->conductivity_coefficients[2]
	<< "\" />\n";
 
      //
      // C11
      FEniCS_xml_C11 
	<< "      <value cell_index=\"" <<  it->cell_id
	<< "\" local_entity=\"0\" value=\""
	<< it->conductivity_coefficients[3]
	<< "\" />\n";
 
      //
      // C12
      FEniCS_xml_C12 
	<< "      <value cell_index=\"" <<  it->cell_id
	<< "\" local_entity=\"0\" value=\""
	<< it->conductivity_coefficients[4]
	<< "\" />\n";
 
      //
      // C22
      FEniCS_xml_C22 
	<< "      <value cell_index=\"" <<  it->cell_id
	<< "\" local_entity=\"0\" value=\""
	<< it->conductivity_coefficients[5]
	<< "\" />\n";
    }


  //
  // End of tetrahedra
  FEniCS_xml_C00 << "    </mesh_value_collection>\n  </mesh_function>\n</dolfin>\n";
  FEniCS_xml_C01 << "    </mesh_value_collection>\n  </mesh_function>\n</dolfin>\n";
  FEniCS_xml_C02 << "    </mesh_value_collection>\n  </mesh_function>\n</dolfin>\n";
  FEniCS_xml_C11 << "    </mesh_value_collection>\n  </mesh_function>\n</dolfin>\n";
  FEniCS_xml_C12 << "    </mesh_value_collection>\n  </mesh_function>\n</dolfin>\n";
  FEniCS_xml_C22 << "    </mesh_value_collection>\n  </mesh_function>\n</dolfin>\n";


  //
  //
  FEniCS_xml_C00.close();
  FEniCS_xml_C01.close();
  FEniCS_xml_C02.close();
  FEniCS_xml_C11.close();
  FEniCS_xml_C12.close();
  FEniCS_xml_C22.close();
}
//
//
//
void 
Domains_build_mesh::Conductivity_matching()
{
#ifdef GPU
  Conductivity_matching_gpu();
#else
  //  Conductivity_matching_classic();
  Conductivity_matching_knn();
#endif
}
//
//
//
void 
Domains_build_mesh::Conductivity_matching_classic()
{
//  //
//  // typedef
//  typedef typename C3t3::Triangulation Triangulation;
//  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
//  typedef typename Triangulation::Vertex_handle Vertex_handle;
//  typedef typename Triangulation::Cell_handle Cell_handle;
//  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
//  typedef typename Triangulation::Point Point_3;
//  //
//  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
//  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
//  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;
  
  //
  // Tetrahedra mapping
  Cell_pmap cell_pmap( mesh_ );

  //
  // Retrieve the transformation matrix and vector from aseg
  Eigen::Matrix< float, 3, 3 > rotation    = (DAp::get_instance())->get_rotation_();
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
  
  //
  // Retrieve voxelization information from conductivity
  int eigenvalues_number_of_pixels_x = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
  int eigenvalues_number_of_pixels_y = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
  int eigenvalues_number_of_pixels_z = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();

  //
  // Retrieve the conductivity data array to match the cell's mesh
  Eigen::Matrix <float, 3, 3>* conductivity_tensors_array  = nullptr;
  Eigen::Matrix <float, 3, 3>* eigen_values_matrices_array = nullptr;
  Eigen::Matrix <float, 3, 1>* positions_array             = nullptr;
  bool*                        Do_we_have_conductivity     = nullptr; 
  //
#ifdef TRACE
#if TRACE == 100
  Eigen::Matrix <float, 3, 3>* P_matrices_array  = nullptr;
  (DAp::get_instance())->get_P_matrices_array_( &P_matrices_array );
#endif
#endif
  (DAp::get_instance())->get_conductivity_tensors_array_( &conductivity_tensors_array );
  (DAp::get_instance())->get_eigen_values_matrices_array_( &eigen_values_matrices_array );
  (DAp::get_instance())->get_positions_array_( &positions_array );
  (DAp::get_instance())->get_Do_we_have_conductivity_( &Do_we_have_conductivity );


  //
  // Main loop
  Point_3 
    CGAL_cell_vertices[5],
    CGAL_cell_centroid;
  Eigen::Matrix< float, 3, 1 > cell_vertices[5];
  //
  int inum = 0; 
  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ;
       cit != mesh_.cells_in_complex_end() ;
       ++cit )
    {
      //
      // link of the linked list list_coefficients_
      Cell_coefficient cell_coeff;
      cell_coeff.cell_id        = inum++;
      cell_coeff.cell_subdomain = cell_pmap.subdomain_index( cit );

#ifdef TRACE
#if TRACE == 4
      if ( inum % 100000 == 0 )
	std::cout << "cell: " << inum << std::endl;
#endif
#endif

      //
      // Vertices positions and centroid of the cell
      // i = 0, 1, 2, 3: VERTICES
      // i = 4 CENTROID
      for (int i = 0 ; i < 4 ; i++)
	{
	  CGAL_cell_vertices[i] = cit->vertex( i )->point();
	  //
	  cell_vertices[i] <<
	    (float)CGAL_cell_vertices[i].x(),
	    (float)CGAL_cell_vertices[i].y(),
	    (float)CGAL_cell_vertices[i].z();
	}
      // centroid
      CGAL_cell_centroid = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);
      cell_vertices[4] <<
	(float)CGAL_cell_centroid.x(),
	(float)CGAL_cell_centroid.y(),
	(float)CGAL_cell_centroid.z();
      // move points from data to framework
      for (int i = 0 ; i < 5 ; i++)
	cell_vertices[i] = rotation * cell_vertices[i] + translation;


      //
      // Output for R analysis
#ifdef TRACE
#if TRACE == 100
      for (int i = 0 ; i < 5 ; i++)
	cell_coeff.vertices[i] = cell_vertices[i];
#endif
#endif      


      //
      // 
      if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION    &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SCALP      &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SKULL      &&
	  cell_pmap.subdomain_index( cit ) != CEREBROSPINAL_FLUID )
	{
	  int 
	    index_val = 0;
	  int 
	    index_min_distance_v[5] = {0,0,0,0,0};
	  float 
	    distance = 0.;
	  float
	    distance_min_v[5] = {100000000.,100000000.,100000000.,100000000.,100000000.};

	  //
	  for ( int dim3 = 0 ; dim3 < eigenvalues_number_of_pixels_z ; dim3++ )
	    for ( int dim2 = 0 ; dim2 < eigenvalues_number_of_pixels_y ; dim2++ )
	      for ( int dim1 = 0 ; dim1 < eigenvalues_number_of_pixels_x ; dim1++ )
		{
		  //
		  // Select the index
		  index_val = dim1 
		    + dim2 * eigenvalues_number_of_pixels_x 
		    + dim3 * eigenvalues_number_of_pixels_x * eigenvalues_number_of_pixels_y;
		  //
		  if( Do_we_have_conductivity[ index_val ] )
		    {
		      //
		      // find the position minimizing the distance with the vertices and centroid
		      for( int i = 0 ; i < 5 ; i++)
			{
			  distance = 
			    /*sqrt(*/ 
			    (positions_array[ index_val ](0) - cell_vertices[i](0)) * 
			    (positions_array[ index_val ](0) - cell_vertices[i](0)) +
			    (positions_array[ index_val ](1) - cell_vertices[i](1)) * 
			    (positions_array[ index_val ](1) - cell_vertices[i](1)) +
			    (positions_array[ index_val ](2) - cell_vertices[i](2)) * 
			    (positions_array[ index_val ](2) - cell_vertices[i](2)) /*)*/;
			  //
			  if( distance < VOXEL_LIMIT)
			    if ( distance < distance_min_v[i]  )
			      {
				distance_min_v[i] = distance;
				index_min_distance_v[i] = index_val;
			      }
			}
		    }
		}


	  //
	  // Cell's conductivity tensor setup
	  int index_min_distance = 0;
	  if ( eigen_values_matrices_array[index_min_distance_v[4]](2,2) > 0. )
	    index_min_distance = index_min_distance_v[4]; /*CENTROID*/
	  else
	    {/* VERTICES*/
	      // select the vertex with positive eigenvalues
	      int tetrahedron_vertex = -1;
	      while( eigen_values_matrices_array[ index_min_distance_v[ ++tetrahedron_vertex ] ](2,2) < 0. )
		if ( tetrahedron_vertex >= 3 )
		  {
		    // we don't have vertices with positive eigenvalues
		    tetrahedron_vertex++;
		    break;
		  }
	      // all the vertices eigenvalues are negatives: switch off the cell
	      if( tetrahedron_vertex < 4 ) /*NO CENTROID NOR VERTICES*/
		index_min_distance = index_min_distance_v[tetrahedron_vertex];
	      else
		index_min_distance = -1;
	    }

	  //
	  //
	  if( index_min_distance != -1 )
	    {/*CENTROID OR VERTICES*/
	      cell_coeff.conductivity_coefficients[0] 
		= conductivity_tensors_array[index_min_distance](0,0);
	      cell_coeff.conductivity_coefficients[1] 
		= conductivity_tensors_array[index_min_distance](0,1);
	      cell_coeff.conductivity_coefficients[2] 	       
		= conductivity_tensors_array[index_min_distance](0,2);
	      cell_coeff.conductivity_coefficients[3] 	       
		= conductivity_tensors_array[index_min_distance](1,1);
	      cell_coeff.conductivity_coefficients[4] 	       
		= conductivity_tensors_array[index_min_distance](1,2);
	      cell_coeff.conductivity_coefficients[5] 	       
		= conductivity_tensors_array[index_min_distance](2,2);

	      //
	      // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	      // l1, l2, l3
	      cell_coeff.eigen_values[0] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
	      cell_coeff.eigen_values[1] = eigen_values_matrices_array[index_min_distance_v[4]](1,1);
	      cell_coeff.eigen_values[2] = eigen_values_matrices_array[index_min_distance_v[4]](2,2);
	      // l_long l_tang l_mean
	      cell_coeff.eigen_values[3] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
	      cell_coeff.eigen_values[4] = (cell_coeff.eigen_values[1]+cell_coeff.eigen_values[2]) / 2.;
	      cell_coeff.eigen_values[5] = (cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0] ) / 3.;
	      // l1_v0 l2_v0 l3_v0 - l1_v1 l2_v1 l3_v1 - l1_v3 l2_v3 l3_v3
	      for ( int i = 0 ; i < 4 ; i++ )
		{
		  cell_coeff.eigen_values[6+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](0,0);
		  cell_coeff.eigen_values[7+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](1,1);
		  cell_coeff.eigen_values[8+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](2,2);
		}
	      //
	      Eigen::Vector3f vec_tmp;
	      vec_tmp <<
		P_matrices_array[index_min_distance](0,0),
		P_matrices_array[index_min_distance](1,0),
		P_matrices_array[index_min_distance](2,0);
	      //
	      cell_coeff.eigenvector_1  = rotation * vec_tmp;
	      cell_coeff.eigenvector_1 /= cell_coeff.eigenvector_1.norm();
	      

#endif
#endif
	    }
	  else
	    {/*NO CENTROID NOR VERTICES*/
	      //
	      // Cell caracteristics are moved in the CSF
	      // Diagonal
	      cell_coeff.conductivity_coefficients[0] = cell_coeff.conductivity_coefficients[3] = cell_coeff.conductivity_coefficients[5] = 1.79;
	      // Non diagonal
	      cell_coeff.conductivity_coefficients[1] = cell_coeff.conductivity_coefficients[2] = cell_coeff.conductivity_coefficients[4] = 0.;
	      
	      //
	      // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	      for ( int i = 0 ; i < 18 ; i++)
		cell_coeff.eigen_values[i] = 0.;
	      //
	      cell_coeff.eigen_values[0] = cell_coeff.eigen_values[4] = cell_coeff.eigen_values[8] = 1.79;
	      //
	      cell_coeff.eigenvector_1 << 0., 0., 0.;
#endif
#endif      
	      
//	      for ( int i = 0 ; i < 6 ; i++)
//		cell_coeff.conductivity_coefficients[i] = 0.;
//	      
//	      //
//	      // Output for R analysis
//#ifdef TRACE
//#if TRACE == 100
//	      for ( int i = 0 ; i < 18 ; i++)
//		cell_coeff.eigen_values[i] = 0.;
//#endif
//#endif
	    }      
	} /*if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION && ... )*/
      else if ( cell_pmap.subdomain_index( cit ) == CEREBROSPINAL_FLUID )
	{
	  // Diagonal
	  cell_coeff.conductivity_coefficients[0] = cell_coeff.conductivity_coefficients[3] = cell_coeff.conductivity_coefficients[5] = 1.79;
	  // Non diagonal
	  cell_coeff.conductivity_coefficients[1] = cell_coeff.conductivity_coefficients[2] = cell_coeff.conductivity_coefficients[4] = 0.;

	  //
	  // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	  for ( int i = 0 ; i < 18 ; i++)
	    cell_coeff.eigen_values[i] = 0.;
	  //
	  cell_coeff.eigen_values[0] = cell_coeff.eigen_values[4] = cell_coeff.eigen_values[8] = 1.79;
	      //
	      cell_coeff.eigenvector_1 << 0., 0., 0.;
#endif
#endif      
	}
      else
	{
	  for ( int i = 0 ; i < 6 ; i++)
	    cell_coeff.conductivity_coefficients[i] = 0.;

	  //
	  // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	  for ( int i = 0 ; i < 18 ; i++)
	    cell_coeff.eigen_values[i] = 0.;
	  //
	  cell_coeff.eigenvector_1 << 0., 0., 0.;
#endif
#endif      
	}
      
      //
      // Add link to the list
      list_coefficients_.push_back( cell_coeff );
    }


  //
  // Output for R analysis
  Conductivity_matching_analysis();


  //
  // Clean up
  delete [] conductivity_tensors_array;
  conductivity_tensors_array = nullptr;
  delete [] eigen_values_matrices_array;
  eigen_values_matrices_array = nullptr;
  delete [] positions_array;
  positions_array = nullptr;
  delete [] Do_we_have_conductivity; 
  Do_we_have_conductivity = nullptr; 
#ifdef TRACE
#if TRACE == 100
  delete [] P_matrices_array;
  P_matrices_array = nullptr;
#endif
#endif
}
//
//
//
void 
Domains_build_mesh::Conductivity_matching_knn()
{
  //
  // Tetrahedra mapping
  Cell_pmap cell_pmap( mesh_ );

  //
  // Retrieve the transformation matrix and vector from aseg
  Eigen::Matrix< float, 3, 3 > rotation    = (DAp::get_instance())->get_rotation_();
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
  
  //
  // Retrieve voxelization information from conductivity
  int eigenvalues_number_of_pixels_x = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
  int eigenvalues_number_of_pixels_y = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
  int eigenvalues_number_of_pixels_z = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();

  //
  // Retrieve the conductivity data array to match the cell's mesh
  Eigen::Matrix <float, 3, 3>* conductivity_tensors_array  = nullptr;
  Eigen::Matrix <float, 3, 3>* eigen_values_matrices_array = nullptr;
  Eigen::Matrix <float, 3, 1>* positions_array             = nullptr;
  bool*                        Do_we_have_conductivity     = nullptr; 
  //
#ifdef TRACE
#if TRACE == 100
  Eigen::Matrix <float, 3, 3>* P_matrices_array  = nullptr;
  (DAp::get_instance())->get_P_matrices_array_( &P_matrices_array );
#endif
#endif
  (DAp::get_instance())->get_conductivity_tensors_array_( &conductivity_tensors_array );
  (DAp::get_instance())->get_eigen_values_matrices_array_( &eigen_values_matrices_array );
  (DAp::get_instance())->get_positions_array_( &positions_array );
  (DAp::get_instance())->get_Do_we_have_conductivity_( &Do_we_have_conductivity );

  //
  // Build the CGAL k-nearest neighbor tree
  Tree tree_conductivity_positions;
  int 
    index_val = 0;
  //
  for ( int dim3 = 0 ; dim3 < eigenvalues_number_of_pixels_z ; dim3++ )
    for ( int dim2 = 0 ; dim2 < eigenvalues_number_of_pixels_y ; dim2++ )
      for ( int dim1 = 0 ; dim1 < eigenvalues_number_of_pixels_x ; dim1++ )
	{
	  //
	  // Select the index
	  index_val = dim1 
	    + dim2 * eigenvalues_number_of_pixels_x 
	    + dim3 * eigenvalues_number_of_pixels_x * eigenvalues_number_of_pixels_y;
	  //
	  if( Do_we_have_conductivity[ index_val ] )
	    tree_conductivity_positions.insert( std::make_tuple( Point_3( positions_array[ index_val ](0),
									  positions_array[ index_val ](1),
									  positions_array[ index_val ](2) ), 
								 index_val) );
	}
  
  //
  // Main loop
  Point_3 
    CGAL_cell_vertices[5],
    CGAL_cell_centroid;
  Eigen::Matrix< float, 3, 1 > cell_vertices[5];
  //
  int inum = 0; 
  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ;
       cit != mesh_.cells_in_complex_end() ;
       ++cit )
    {
      //
      // link of the linked list list_coefficients_
      Cell_coefficient cell_coeff;
      cell_coeff.cell_id        = inum++;
      cell_coeff.cell_subdomain = cell_pmap.subdomain_index( cit );

//#ifdef TRACE
//#if TRACE == 4
      if ( inum % 100000 == 0 )
	std::cout << "cell: " << inum << std::endl;
//#endif
//#endif

      //
      // Vertices positions and centroid of the cell
      // i = 0, 1, 2, 3: VERTICES
      // i = 4 CENTROID
      for (int i = 0 ; i < 4 ; i++)
	{
	  CGAL_cell_vertices[i] = cit->vertex( i )->point();
	  //
	  cell_vertices[i] <<
	    (float)CGAL_cell_vertices[i].x(),
	    (float)CGAL_cell_vertices[i].y(),
	    (float)CGAL_cell_vertices[i].z();
	}
      // centroid
      CGAL_cell_centroid = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);
      cell_vertices[4] <<
	(float)CGAL_cell_centroid.x(),
	(float)CGAL_cell_centroid.y(),
	(float)CGAL_cell_centroid.z();
      // move points from data to framework
      for (int i = 0 ; i < 5 ; i++)
	cell_vertices[i] = rotation * cell_vertices[i] + translation;


      //
      // Output for R analysis
#ifdef TRACE
#if TRACE == 100
      for (int i = 0 ; i < 5 ; i++)
	cell_coeff.vertices[i] = cell_vertices[i];
#endif
#endif      

      //
      // If we are in the brain
      if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION    &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SCALP      &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SKULL      &&
	  cell_pmap.subdomain_index( cit ) != CEREBROSPINAL_FLUID )
	{
	  //
	  // Search the K-nearest neighbor
	  K_neighbor_search search( tree_conductivity_positions, 
				    Point_3( cell_vertices[4/* centroid */](0),
					     cell_vertices[4/* centroid */](1),
					     cell_vertices[4/* centroid */](2) ), 
				    /* K = */ 15);
	  // Get the iterator on the nearest neighbor
	  auto conductivity_centroids = search.begin();

	  //
	  //
	  Distance tr_dist;
	  //
	  for( auto centroid : search )
	    std::cout << " d(q, nearest neighbor)=  "
		      << tr_dist.inverse_of_transformed_distance(centroid.second) 
		      << " -- pos: "   << std::get<0>(centroid.first)
		      << " -- index: " << std::get<1>(centroid.first) 
		      << std::endl;

	  //
	  // 
	  while( conductivity_centroids != search.end() &&
		 eigen_values_matrices_array[std::get<1>( conductivity_centroids->first )](2,2) < 0. )
	    {
	      conductivity_centroids++;
	      std::cout << " d(q, nearest neighbor in while)=  "
			<< tr_dist.inverse_of_transformed_distance(conductivity_centroids->second) 
			<< " -- pos: "   << std::get<0>(conductivity_centroids->first)
			<< " -- index: " << std::get<1>(conductivity_centroids->first) 
			<< " -- l3: " << eigen_values_matrices_array[std::get<1>( conductivity_centroids->first )](2,2) 
			<< std::endl;
	    }
	  //
	  if( conductivity_centroids == search.end() )
	    {
	      std::cerr << "You might think about increasing the number of neighbor. Or check the Diffusion/Conductivity file." << std::endl;
	      exit(1);
	    }
	  //
	  std::cout << " d(q, nearest neighbor after while)=  "
		    << tr_dist.inverse_of_transformed_distance(conductivity_centroids->second) 
		    << " -- pos: "   << std::get<0>(conductivity_centroids->first)
		    << " -- index: " << std::get<1>(conductivity_centroids->first) 
		    << " -- l3: " << eigen_values_matrices_array[std::get<1>( conductivity_centroids->first )](2,2) 
		    << std::endl;
	  //
	  cell_coeff.conductivity_coefficients[0] 
	    = conductivity_tensors_array[std::get<1>( conductivity_centroids->first )](0,0);
	  cell_coeff.conductivity_coefficients[1] 
	    = conductivity_tensors_array[std::get<1>( conductivity_centroids->first )](0,1);
	  cell_coeff.conductivity_coefficients[2] 	       
	    = conductivity_tensors_array[std::get<1>( conductivity_centroids->first )](0,2);
	  cell_coeff.conductivity_coefficients[3] 	       
	    = conductivity_tensors_array[std::get<1>( conductivity_centroids->first )](1,1);
	  cell_coeff.conductivity_coefficients[4] 	       
	    = conductivity_tensors_array[std::get<1>( conductivity_centroids->first )](1,2);
	  cell_coeff.conductivity_coefficients[5] 	       
	    = conductivity_tensors_array[std::get<1>( conductivity_centroids->first )](2,2);
	  
	  //
	  // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	  // l1, l2, l3
	  cell_coeff.eigen_values[0] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
	  cell_coeff.eigen_values[1] = eigen_values_matrices_array[index_min_distance_v[4]](1,1);
	  cell_coeff.eigen_values[2] = eigen_values_matrices_array[index_min_distance_v[4]](2,2);
	  // l_long l_tang l_mean
	  cell_coeff.eigen_values[3] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
	  cell_coeff.eigen_values[4] = (cell_coeff.eigen_values[1]+cell_coeff.eigen_values[2]) / 2.;
	  cell_coeff.eigen_values[5] = (cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0] ) / 3.;
	  // l1_v0 l2_v0 l3_v0 - l1_v1 l2_v1 l3_v1 - l1_v3 l2_v3 l3_v3
	  for ( int i = 0 ; i < 4 ; i++ )
	    {
	      cell_coeff.eigen_values[6+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](0,0);
	      cell_coeff.eigen_values[7+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](1,1);
	      cell_coeff.eigen_values[8+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](2,2);
	    }
	  //
	  Eigen::Vector3f vec_tmp;
	  vec_tmp <<
	    P_matrices_array[index_min_distance](0,0),
	    P_matrices_array[index_min_distance](1,0),
	    P_matrices_array[index_min_distance](2,0);
	  //
	  cell_coeff.eigenvector_1  = rotation * vec_tmp;
	  cell_coeff.eigenvector_1 /= cell_coeff.eigenvector_1.norm();
#endif
#endif
	} /*if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION && ... )*/
      else if ( cell_pmap.subdomain_index( cit ) == CEREBROSPINAL_FLUID )
	{
	  // Diagonal
	  cell_coeff.conductivity_coefficients[0] = cell_coeff.conductivity_coefficients[3] = cell_coeff.conductivity_coefficients[5] = 1.79;
	  // Non diagonal
	  cell_coeff.conductivity_coefficients[1] = cell_coeff.conductivity_coefficients[2] = cell_coeff.conductivity_coefficients[4] = 0.;

	  //
	  // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	  for ( int i = 0 ; i < 18 ; i++)
	    cell_coeff.eigen_values[i] = 0.;
	  //
	  cell_coeff.eigen_values[0] = cell_coeff.eigen_values[4] = cell_coeff.eigen_values[8] = 1.79;
	      //
	      cell_coeff.eigenvector_1 << 0., 0., 0.;
#endif
#endif      
	} /*else if ( cell_pmap.subdomain_index( cit ) == CEREBROSPINAL_FLUID )*/
  // Skull and scalp
      else
	{
	  for ( int i = 0 ; i < 6 ; i++)
	    cell_coeff.conductivity_coefficients[i] = 0.;

	  //
	  // Output for R analysis
#ifdef TRACE
#if TRACE == 100
	  for ( int i = 0 ; i < 18 ; i++)
	    cell_coeff.eigen_values[i] = 0.;
	  //
	  cell_coeff.eigenvector_1 << 0., 0., 0.;
#endif
#endif      
	}
      
      //
      // Add link to the list
      list_coefficients_.push_back( cell_coeff );
    }


  //
  // Output for R analysis
  Conductivity_matching_analysis();


  //
  // Clean up
  delete [] conductivity_tensors_array;
  conductivity_tensors_array = nullptr;
  delete [] eigen_values_matrices_array;
  eigen_values_matrices_array = nullptr;
  delete [] positions_array;
  positions_array = nullptr;
  delete [] Do_we_have_conductivity; 
  Do_we_have_conductivity = nullptr; 
#ifdef TRACE
#if TRACE == 100
  delete [] P_matrices_array;
  P_matrices_array = nullptr;
#endif
#endif
}
//
//
//
void 
Domains_build_mesh::Conductivity_matching_test()
{
//  //
//  // typedef
//  typedef typename C3t3::Triangulation Triangulation;
//  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
//  typedef typename Triangulation::Vertex_handle Vertex_handle;
//  typedef typename Triangulation::Cell_handle Cell_handle;
//  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
//  typedef typename Triangulation::Point Point_3;
//  //
//  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
//  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
//  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;
//  
//  //
//  // Tetrahedra mapping
//  Cell_pmap cell_pmap( mesh_ );
//
//  //
//  // Retrieve the transformation matrix and vector from aseg
//  Eigen::Matrix< float, 3, 3 > rotation    = (DAp::get_instance())->get_rotation_();
//  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
//  
//  //
//  // Retrieve voxelization information from conductivity
//  int eigenvalues_number_of_pixels_x = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
//  int eigenvalues_number_of_pixels_y = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
//  int eigenvalues_number_of_pixels_z = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();
//
//  //
//  // Retrieve the conductivity data array to match the cell's mesh
//  Eigen::Matrix <float, 3, 3>* conductivity_tensors_array  = nullptr;
//  Eigen::Matrix <float, 3, 3>* eigen_values_matrices_array = nullptr;
//  Eigen::Matrix <float, 3, 1>* positions_array             = nullptr;
//  bool*                        Do_we_have_conductivity     = nullptr; 
//  //
//#ifdef TRACE
//#if TRACE == 100
//  Eigen::Matrix <float, 3, 3>* P_matrices_array  = nullptr;
//  (DAp::get_instance())->get_P_matrices_array_( &P_matrices_array );
//#endif
//#endif
//  (DAp::get_instance())->get_conductivity_tensors_array_( &conductivity_tensors_array );
//  (DAp::get_instance())->get_eigen_values_matrices_array_( &eigen_values_matrices_array );
//  (DAp::get_instance())->get_positions_array_( &positions_array );
//  (DAp::get_instance())->get_Do_we_have_conductivity_( &Do_we_have_conductivity );
//
//
//  Tree 
//    tree,
//    tree1;
//  int 
//    index_val = 0;
//  int 
//    index_min_distance_v[5] = {0,0,0,0,0};
//  float 
//    distance = 0.;
//  float
//    distance_min_v[5] = {100000000.,100000000.,100000000.,100000000.,100000000.};
//  
//
//  //
//  for ( int dim3 = 0 ; dim3 < eigenvalues_number_of_pixels_z ; dim3++ )
//    for ( int dim2 = 0 ; dim2 < eigenvalues_number_of_pixels_y ; dim2++ )
//      for ( int dim1 = 0 ; dim1 < eigenvalues_number_of_pixels_x ; dim1++ )
//	{
//	  //
//	  // Select the index
//	  index_val = dim1 
//	    + dim2 * eigenvalues_number_of_pixels_x 
//	    + dim3 * eigenvalues_number_of_pixels_x * eigenvalues_number_of_pixels_y;
//	  //
//	  if( Do_we_have_conductivity[ index_val ] )
//	    {
//	      std::cout 
//		<< (double)positions_array[ index_val ](0) << " "
//		<< (double)positions_array[ index_val ](1) << " "
//		<< (double)positions_array[ index_val ](2) << " "
//		<< index_val << std::endl;
//		tree.insert( Point_dd( positions_array[ index_val ](0),
//				       positions_array[ index_val ](1),
//				       positions_array[ index_val ](2),
//				       index_val ) );
//		tree1.insert( Point_dd( positions_array[ index_val ](0),
//					positions_array[ index_val ](1),
//					positions_array[ index_val ](2) ) );
//	    }
//	}
//  
//
//
//
//  //
//  // Main loop
//  Point_3 
//    CGAL_cell_vertices[5],
//    CGAL_cell_centroid;
//  Eigen::Matrix< float, 3, 1 > cell_vertices[5];
//  //
//  int inum = 0; 
//  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ;
//       cit != mesh_.cells_in_complex_end() ;
//       ++cit )
//    {
//      //
//      // link of the linked list list_coefficients_
//      Cell_coefficient cell_coeff;
//      cell_coeff.cell_id        = inum++;
//      cell_coeff.cell_subdomain = cell_pmap.subdomain_index( cit );
//
//#ifdef TRACE
//#if TRACE == 4
//      if ( inum % 100000 == 0 )
//	std::cout << "cell: " << inum << std::endl;
//#endif
//#endif
//
//      //
//      // Vertices positions and centroid of the cell
//      // Vertices are in the inrimage date frame. They have to be moved in the general data taking framework
//      // i = 0, 1, 2, 3: VERTICES
//      // i = 4 CENTROID
//      for (int i = 0 ; i < 4 ; i++)
//	{
//	  CGAL_cell_vertices[i] = cit->vertex( i )->point();
//	  //
//	  cell_vertices[i] <<
//	    (float)CGAL_cell_vertices[i].x(),
//	    (float)CGAL_cell_vertices[i].y(),
//	    (float)CGAL_cell_vertices[i].z();
//	}
//      // centroid
//      CGAL_cell_centroid = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);
//      cell_vertices[4] <<
//	(float)CGAL_cell_centroid.x(),
//	(float)CGAL_cell_centroid.y(),
//	(float)CGAL_cell_centroid.z();
//      // move points from data to framework
//      for (int i = 0 ; i < 5 ; i++)
//	cell_vertices[i] = rotation * cell_vertices[i] + translation;
//
//
//
//
//
//
//      //
//      // Output for R analysis
//#ifdef TRACE
//#if TRACE == 100
//      for (int i = 0 ; i < 5 ; i++)
//	cell_coeff.vertices[i] = cell_vertices[i];
//#endif
//#endif      
//
//
//      //
//      // Selection of the brain sub-domains
//      if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION    &&
//	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SCALP      &&
//	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SKULL      &&
//	  cell_pmap.subdomain_index( cit ) != CEREBROSPINAL_FLUID )
//	{
//
////      std::cout << "Ref point: " 
////		<< cit->vertex( 0 )->point().x() << " " 
////		<< cit->vertex( 0 )->point().y() << " " 
////		<< cit->vertex( 0 )->point().z() << " - Ref point: " << cit->vertex( 0 )->point() << std::endl;
//
//      Point_dd query( (double)cit->vertex( 0 )->point().x(),
//		      (double)cit->vertex( 0 )->point().y(),
//		      (double)cit->vertex( 0 )->point().z(), 0 );
//      NN_incremental_search NN(tree, query);
//      NN_incremental_search NN1(tree1, query);
//
//      std::cout <<  "The first 5 nearest neighbours with positive x-coord are: " << std::endl;
//      NN_incremental_search::iterator it = NN.begin();
//      for (int j = 0; (j < 5) && (it != NN.end() ) ; ++j, it++)
//	std::cout << "point ref: " << cit->vertex( 0 )->point() << " --- " 
//		  << (*it).first << "  at squared distance = " << (*it).second << std::endl;
//      std::cout <<  "The first 5 nearest neighbours with positive x-coord are: " << std::endl;
//
//      NN_incremental_search::iterator it1 = NN1.begin();
//      for (int j = 0; (j < 5) && (it1 != NN1.end() ) ; ++j, it1++)
//	std::cout << "point ref: " << cit->vertex( 0 )->point() << " --- " 
//		  << (*it1).first << "  at squared distance = " << (*it1).second << std::endl;
//
//
////
////	  //
////	  // Cell's conductivity tensor setup
////	  int index_min_distance = 0;
////	  if ( eigen_values_matrices_array[index_min_distance_v[4]](2,2) > 0. )
////	    index_min_distance = index_min_distance_v[4]; /*CENTROID*/
////	  else
////	    {/* VERTICES*/
////	      // select the vertex with positive eigenvalues
////	      int tetrahedron_vertex = -1;
////	      while( eigen_values_matrices_array[ index_min_distance_v[ ++tetrahedron_vertex ] ](2,2) < 0. )
////		if ( tetrahedron_vertex >= 3 )
////		  {
////		    // we don't have vertices with positive eigenvalues
////		    tetrahedron_vertex++;
////		    break;
////		  }
////	      // all the vertices eigenvalues are negatives: switch off the cell
////	      if( tetrahedron_vertex < 4 ) /*NO CENTROID NOR VERTICES*/
////		index_min_distance = index_min_distance_v[tetrahedron_vertex];
////	      else
////		index_min_distance = -1;
////	    }
////
////	  //
////	  //
////	  if( index_min_distance != -1 )
////	    {/*CENTROID OR VERTICES*/
////	      cell_coeff.conductivity_coefficients[0] 
////		= conductivity_tensors_array[index_min_distance](0,0);
////	      cell_coeff.conductivity_coefficients[1] 
////		= conductivity_tensors_array[index_min_distance](0,1);
////	      cell_coeff.conductivity_coefficients[2] 	       
////		= conductivity_tensors_array[index_min_distance](0,2);
////	      cell_coeff.conductivity_coefficients[3] 	       
////		= conductivity_tensors_array[index_min_distance](1,1);
////	      cell_coeff.conductivity_coefficients[4] 	       
////		= conductivity_tensors_array[index_min_distance](1,2);
////	      cell_coeff.conductivity_coefficients[5] 	       
////		= conductivity_tensors_array[index_min_distance](2,2);
////
////	      //
////	      // Output for R analysis
////#ifdef TRACE
////#if TRACE == 100
////	      // l1, l2, l3
////	      cell_coeff.eigen_values[0] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
////	      cell_coeff.eigen_values[1] = eigen_values_matrices_array[index_min_distance_v[4]](1,1);
////	      cell_coeff.eigen_values[2] = eigen_values_matrices_array[index_min_distance_v[4]](2,2);
////	      // l_long l_tang l_mean
////	      cell_coeff.eigen_values[3] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
////	      cell_coeff.eigen_values[4] = (cell_coeff.eigen_values[1]+cell_coeff.eigen_values[2]) / 2.;
////	      cell_coeff.eigen_values[5] = (cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0] ) / 3.;
////	      // l1_v0 l2_v0 l3_v0 - l1_v1 l2_v1 l3_v1 - l1_v3 l2_v3 l3_v3
////	      for ( int i = 0 ; i < 4 ; i++ )
////		{
////		  cell_coeff.eigen_values[6+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](0,0);
////		  cell_coeff.eigen_values[7+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](1,1);
////		  cell_coeff.eigen_values[8+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](2,2);
////		}
////	      //
////	      Eigen::Vector3f vec_tmp;
////	      vec_tmp <<
////		P_matrices_array[index_min_distance](0,0),
////		P_matrices_array[index_min_distance](1,0),
////		P_matrices_array[index_min_distance](2,0);
////	      //
////	      cell_coeff.eigenvector_1  = rotation * vec_tmp;
////	      cell_coeff.eigenvector_1 /= cell_coeff.eigenvector_1.norm();
////	      
////
////#endif
////#endif
////	    }
////	  else
////	    {/*NO CENTROID NOR VERTICES*/
////	      //
////	      // Cell caracteristics are moved in the CSF
////	      // Diagonal
////	      cell_coeff.conductivity_coefficients[0] = cell_coeff.conductivity_coefficients[3] = cell_coeff.conductivity_coefficients[5] = 1.79;
////	      // Non diagonal
////	      cell_coeff.conductivity_coefficients[1] = cell_coeff.conductivity_coefficients[2] = cell_coeff.conductivity_coefficients[4] = 0.;
////	      
////	      //
////	      // Output for R analysis
////#ifdef TRACE
////#if TRACE == 100
////	      for ( int i = 0 ; i < 18 ; i++)
////		cell_coeff.eigen_values[i] = 0.;
////	      //
////	      cell_coeff.eigen_values[0] = cell_coeff.eigen_values[4] = cell_coeff.eigen_values[8] = 1.79;
////	      //
////	      cell_coeff.eigenvector_1 << 0., 0., 0.;
////#endif
////#endif      
////	      
//////	      for ( int i = 0 ; i < 6 ; i++)
//////		cell_coeff.conductivity_coefficients[i] = 0.;
//////	      
//////	      //
//////	      // Output for R analysis
//////#ifdef TRACE
//////#if TRACE == 100
//////	      for ( int i = 0 ; i < 18 ; i++)
//////		cell_coeff.eigen_values[i] = 0.;
//////#endif
//////#endif
////	    }      
////	} /*if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION && ... )*/
////      else if ( cell_pmap.subdomain_index( cit ) == CEREBROSPINAL_FLUID )
////	{
////	  // Diagonal
////	  cell_coeff.conductivity_coefficients[0] = cell_coeff.conductivity_coefficients[3] = cell_coeff.conductivity_coefficients[5] = 1.79;
////	  // Non diagonal
////	  cell_coeff.conductivity_coefficients[1] = cell_coeff.conductivity_coefficients[2] = cell_coeff.conductivity_coefficients[4] = 0.;
////
////	  //
////	  // Output for R analysis
////#ifdef TRACE
////#if TRACE == 100
////	  for ( int i = 0 ; i < 18 ; i++)
////	    cell_coeff.eigen_values[i] = 0.;
////	  //
////	  cell_coeff.eigen_values[0] = cell_coeff.eigen_values[4] = cell_coeff.eigen_values[8] = 1.79;
////	      //
////	      cell_coeff.eigenvector_1 << 0., 0., 0.;
////#endif
////#endif      
////	}
////      else
////	{
////	  for ( int i = 0 ; i < 6 ; i++)
////	    cell_coeff.conductivity_coefficients[i] = 0.;
////
////	  //
////	  // Output for R analysis
////#ifdef TRACE
////#if TRACE == 100
////	  for ( int i = 0 ; i < 18 ; i++)
////	    cell_coeff.eigen_values[i] = 0.;
////	  //
////	  cell_coeff.eigenvector_1 << 0., 0., 0.;
////#endif
////#endif      
//	}
//      
//      //
//      // Add link to the list
//      list_coefficients_.push_back( cell_coeff );
//    }
//
//
//  //
//  // Output for R analysis
//  Conductivity_matching_analysis();
//
//
//  //
//  // Clean up
//  delete [] conductivity_tensors_array;
//  conductivity_tensors_array = nullptr;
//  delete [] eigen_values_matrices_array;
//  eigen_values_matrices_array = nullptr;
//  delete [] positions_array;
//  positions_array = nullptr;
//  delete [] Do_we_have_conductivity; 
//  Do_we_have_conductivity = nullptr; 
//#ifdef TRACE
//#if TRACE == 100
//  delete [] P_matrices_array;
//  P_matrices_array = nullptr;
//#endif
//#endif
}
//
//
//
void 
Domains_build_mesh::Conductivity_matching_gpu()
{
//  //
//  // typedef
//  typedef typename C3t3::Triangulation Triangulation;
//  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
//  typedef typename Triangulation::Vertex_handle Vertex_handle;
//  typedef typename Triangulation::Cell_handle Cell_handle;
//  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
//  typedef typename Triangulation::Point Point_3;
//  //
//  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
//  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
//  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;

  //
  // Tetrahedra mapping
  Cell_pmap cell_pmap( mesh_ );


  //
  // Retrieve the transformation matrix and vector from aseg
  Eigen::Matrix< float, 3, 3 > rotation    = (DAp::get_instance())->get_rotation_();
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
  
  //
  // Retrieve voxelization information from conductivity
  int eigenvalues_number_of_pixels_x = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
  int eigenvalues_number_of_pixels_y = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
  int eigenvalues_number_of_pixels_z = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();

  //
  // Retrieve the conductivity data array to match the cell's mesh
  Eigen::Matrix <float, 3, 3>* conductivity_tensors_array  = nullptr;
  Eigen::Matrix <float, 3, 3>* eigen_values_matrices_array = nullptr;
  Eigen::Matrix <float, 3, 1>* positions_array             = nullptr;
  bool*                        Do_we_have_conductivity     = nullptr; 
  //
  (DAp::get_instance())->get_conductivity_tensors_array_( &conductivity_tensors_array );
  (DAp::get_instance())->get_eigen_values_matrices_array_( &eigen_values_matrices_array );
  (DAp::get_instance())->get_positions_array_( &positions_array );
  (DAp::get_instance())->get_Do_we_have_conductivity_( &Do_we_have_conductivity );

  //
  // Prepare CUDA array
  int 
    array_size = eigenvalues_number_of_pixels_x * eigenvalues_number_of_pixels_y * eigenvalues_number_of_pixels_z;
  float
    voxel_center_position_x[array_size],
    voxel_center_position_y[array_size],
    voxel_center_position_z[array_size];
  //
  // distance_array is the minimum distance for each point of the tetrahedron from the centers of voxels. 
  // distance_index_array is the minimum distance index for each point of the tetrahedron. 
  // In other words, it represents the voxel index of each point of the tetrahedron
  // [distance_index_array] =  BLOCKS * 5. It collect the smallest distance among the THREADS 
  // for each BLOCK.
  // | min_dist_v0 | min_dist_v1 | min_dist_v2 | min_dist_v3 | min_dist_C | min_dist_v0  | min_dist_v1 | ...
  // |                         BLOCK0                                     |                   BLOCK1 ...
  // 
  float distance_array[5*(BLOCKS+REMAIN)];
  int   distance_index_array[5*(BLOCKS+REMAIN)];

  //
  for ( int i = 0 ; 
	i < array_size;
	i++ )
    {
      voxel_center_position_x[i] = positions_array[i](0);
      voxel_center_position_y[i] = positions_array[i](1);
      voxel_center_position_z[i] = positions_array[i](2);
    }
  // we do not need positions_array anymore
  delete [] positions_array;
  positions_array = nullptr;

  //
  // Intitialisation of the CUDA data
  CUDA_Conductivity_matching cuda_matcher( array_size,
					   voxel_center_position_x,
					   voxel_center_position_y,
					   voxel_center_position_z,
					   Do_we_have_conductivity);


  //
  // Main loop
  Point_3 
    CGAL_cell_vertices[5],
    CGAL_cell_centroid;
  Eigen::Matrix< float, 3, 1 > cell_vertices[5];
  //
  std::cout << "cell: Starting" << std::endl;
  int inum = 0; 

  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ; cit != mesh_.cells_in_complex_end() ; ++cit )
    {
      //
      // link of the linked list list_coefficients_
      Cell_coefficient cell_coeff;
      cell_coeff.cell_id        = inum++;
      cell_coeff.cell_subdomain = cell_pmap.subdomain_index( cit );
      //
#ifdef TRACE
#if TRACE == 4
      if ( inum % 100000 == 0 )
	std::cout << "cell: " << inum << std::endl;
#endif
#endif

      //
      // Vertices positions and centroid of the cell
      // i = 0, 1, 2, 3: VERTICES
      // i = 4 CENTROID
      for (int i = 0 ; i < 4 ; i++)
	{
	  CGAL_cell_vertices[i] = cit->vertex( i )->point();
	  //
	  cell_vertices[i] <<
	    (float)CGAL_cell_vertices[i].x(),
	    (float)CGAL_cell_vertices[i].y(),
	    (float)CGAL_cell_vertices[i].z();
	}
      // centroid
      CGAL_cell_centroid = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);
      cell_vertices[4] <<
	(float)CGAL_cell_centroid.x(),
	(float)CGAL_cell_centroid.y(),
	(float)CGAL_cell_centroid.z();
      // move points from data to framework
      for (int i = 0 ; i < 5 ; i++)
	cell_vertices[i] = rotation * cell_vertices[i] + translation;


      //
      // Output for R analysis
#ifdef TRACE
#if TRACE == 100
      for (int i = 0 ; i < 5 ; i++)
	cell_coeff.vertices[i] = cell_vertices[i];
#endif
#endif      


      //
      // 
      if( cell_coeff.cell_subdomain != NO_SEGMENTATION     &&
	  cell_coeff.cell_subdomain != OUTSIDE_SCALP      &&
	  cell_coeff.cell_subdomain != OUTSIDE_SKULL      &&
	  cell_coeff.cell_subdomain != CEREBROSPINAL_FLUID )
	{

	  //
	  // Linearization of vertices coordinates
	  float cell_points[15];
	  //
	  for (int i = 0 ; i < 5 ; i ++ )
	    {
	      cell_points[3*i + 0] = cell_vertices[i](0); 
	      cell_points[3*i + 1] = cell_vertices[i](1); 
	      cell_points[3*i + 2] = cell_vertices[i](2); 
	    }

	  //
	  // launch the CUDA kernel
	  cuda_matcher.find_vertices_voxel_index( cell_points, 
						  distance_array, 
						  distance_index_array );


	  //
	  //
	  int 
	    index_min_distance_v[5] = {0,0,0,0,0};
	  float
	    distance_min_v[5] = {100000000.,100000000.,100000000.,100000000.,100000000.};
	  //
	  for ( int block = 0 ; block < (BLOCKS+REMAIN) ; block++ )
	    {
	      for ( int v = 0 ; v < 5 ; v++ )
		{
		  if( distance_array[5*block + v] < distance_min_v[v] )
		    {
		      distance_min_v[v]       = distance_array[5*block + v];
		      index_min_distance_v[v] = distance_index_array[5*block + v];
		    }
		}
	    }
	  
	  
	  //	  std::cout << std::endl;
	  //	  std::cout << "###################" << std::endl;
	  //	  for ( int v = 0 ; v < 5 ; v++ )
	  //	    std::cout << "index_v[" << v << "] =  " << index_min_distance_v[v] 
	  //		      << " - distance_v[" << v << "] = " << distance_min_v[v] << std::endl;

	  //	  int 
	  //	    index_val = 0;
	  //	  int 
	  //	    index_min_distance_v[5] = {0,0,0,0,0};
	  //	  float 
	  //	    distance = 0.;
	  //	  float
	  //	    distance_min_v[5] = {100000000.,100000000.,100000000.,100000000.,100000000.};

	  //	  //
	  //	  for ( int dim3 = 0 ; dim3 < eigenvalues_number_of_pixels_z ; dim3++ )
	  //	    for ( int dim2 = 0 ; dim2 < eigenvalues_number_of_pixels_y ; dim2++ )
	  //	      for ( int dim1 = 0 ; dim1 < eigenvalues_number_of_pixels_x ; dim1++ )
	  //		{
	  //		}
	  //
	  //
	  //	  //
	  //	  // Cell's conductivity tensor setup
	  //	  int index_min_distance = 0;
	  //	  if ( eigen_values_matrices_array[index_min_distance_v[4]](2,2) > 0. )
	  //	    index_min_distance = index_min_distance_v[4]; /*CENTROID*/
	  //	  else
	  //	    {/* VERTICES*/
	  //	      // select the vertex with positive eigenvalues
	  //	      int tetrahedron_vertex = -1;
	  //	      while( eigen_values_matrices_array[ index_min_distance_v[ ++tetrahedron_vertex ] ](2,2) < 0. )
	  //		if ( tetrahedron_vertex >= 3 )
	  //		  {
	  //		    // we don't have vertices with positiv eigenvalues
	  //		    tetrahedron_vertex++;
	  //		    break;
	  //		  }
	  //	      // all the vertices eigenvalues are negatives: switch off the cell
	  //	      if( tetrahedron_vertex < 4 ) /*NO CENTROID NOR VERTICES*/
	  //		index_min_distance = index_min_distance_v[tetrahedron_vertex];
	  //	      else
	  //		index_min_distance = -1;
	  //	    }
	  //
	  //	  //
	  //	  //
	  //	  if( index_min_distance != -1 )
	  //	    {/*CENTROID OR VERTICES*/
	  //	      cell_coeff.conductivity_coefficients[0] 
	  //		= conductivity_tensors_array[index_min_distance](0,0);
	  //	      cell_coeff.conductivity_coefficients[1] 
	  //		= conductivity_tensors_array[index_min_distance](0,1);
	  //	      cell_coeff.conductivity_coefficients[2] 	       
	  //		= conductivity_tensors_array[index_min_distance](0,2);
	  //	      cell_coeff.conductivity_coefficients[3] 	       
	  //		= conductivity_tensors_array[index_min_distance](1,1);
	  //	      cell_coeff.conductivity_coefficients[4] 	       
	  //		= conductivity_tensors_array[index_min_distance](1,2);
	  //	      cell_coeff.conductivity_coefficients[5] 	       
	  //		= conductivity_tensors_array[index_min_distance](2,2);
	  //
	  //	      //
	  //	      // Output for R analysis
	  //#ifdef TRACE
	  //#if TRACE == 100
	  //	      // l1, l2, l3
	  //	      cell_coeff.eigen_values[0] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
	  //	      cell_coeff.eigen_values[1] = eigen_values_matrices_array[index_min_distance_v[4]](1,1);
	  //	      cell_coeff.eigen_values[2] = eigen_values_matrices_array[index_min_distance_v[4]](2,2);
	  //	      // l_long l_tang l_mean
	  //	      cell_coeff.eigen_values[3] = eigen_values_matrices_array[index_min_distance_v[4]](0,0);
	  //	      cell_coeff.eigen_values[4] = (cell_coeff.eigen_values[1]+cell_coeff.eigen_values[2]) / 2.;
	  //	      cell_coeff.eigen_values[5] = (cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0]+cell_coeff.eigen_values[0] ) / 3.;
	  //	      // l1_v0 l2_v0 l3_v0 - l1_v1 l2_v1 l3_v1 - l1_v3 l2_v3 l3_v3
	  //	      for ( int i = 0 ; i < 4 ; i++ )
	  //		{
	  //		  cell_coeff.eigen_values[6+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](0,0);
	  //		  cell_coeff.eigen_values[7+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](1,1);
	  //		  cell_coeff.eigen_values[8+i*3] = eigen_values_matrices_array[index_min_distance_v[i]](2,2);
	  //		}
	  //#endif
	  //#endif
	  //	    }
	  //	  else
	  //	    {/*NO CENTROID NOR VERTICES*/
	  //	      for ( int i = 0 ; i < 5 ; i++)
	  //		cell_coeff.conductivity_coefficients[i] = 0.;
	  //	      
	  //	      //
	  //	      // Output for R analysis
	  //#ifdef TRACE
	  //#if TRACE == 100
	  //	      for ( int i = 0 ; i < 18 ; i++)
	  //		cell_coeff.eigen_values[i] = 0.;
	  //#endif
	  //#endif
	  //	    } 
	  //
	  //	  
	} /*if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION && ... )*/
      else
	{
	  //	  for ( int i = 0 ; i < 5 ; i++)
	  //	    cell_coeff.conductivity_coefficients[i] = 0.;
	  //
	  //	  //
	  //	  // Output for R analysis
	  //ifdef TRACE
	  //if TRACE == 100
	  //	  for ( int i = 0 ; i < 18 ; i++)
	  //	    cell_coeff.eigen_values[i] = 0.;
	  //endif
	  //endif      
	}
      
      //      //
      //      // Add link to the list
      //      list_coefficients_.push_back( cell_coeff );
    } /*for( Cell_iterator cit = mesh_.cells_in_complex_begin() ; cit != mesh_.cells_in_complex_end() ; ++cit )*/


  //
  // Output for R analysis
  Conductivity_matching_analysis();


  //
  // Clean up
  delete [] conductivity_tensors_array;
  conductivity_tensors_array = nullptr;
  delete [] eigen_values_matrices_array;
  eigen_values_matrices_array = nullptr;
  delete [] Do_we_have_conductivity; 
  Do_we_have_conductivity = nullptr; 
}
//
//
//
void
Domains_build_mesh::Conductivity_matching_analysis()
{
#ifdef TRACE
#if TRACE == 100
  //
  // Stream
  std::stringstream 
    err,
    err1;
  //
  err 
    << "Cell_sub_domain "
    << "X_cent Y_cent Z_cent  "
    << "l1  l2  l3 l_long l_tang l_mean "
    << "l1_v0 l2_v0 l3_v0 "
    << "l1_v1 l2_v1 l3_v1 "
    << "l1_v2 l2_v2 l3_v2 "
    << "l1_v3 l2_v3 l3_v3 \n";
  //
  err1 
    << "Cell_sub_domain "
    << "X Y Z "
    << "v11 v12 v13 \n";


  //
  // Main loop
  for ( auto it = list_coefficients_.begin() ; 
	it != list_coefficients_.end() ; 
	++it )
    {
      err 
	<< it->cell_subdomain << " "
	<< it->vertices[4](0) << " " << it->vertices[4](1) << " " << it->vertices[4](2) << " ";
      //
      err1 
	<< it->cell_subdomain << " "
	<< it->vertices[4](0) << " " << it->vertices[4](1) << " " << it->vertices[4](2) << " "
	<< it->eigenvector_1(0) << " " << it->eigenvector_1(1) << " " << it->eigenvector_1(2) << " ";
      //
      for( int i = 0 ; i < 18 ; i++ )
	err << it->eigen_values[i] << " ";
      //
      err << std::endl;
      err1 << std::endl;
    }


  //
  // 
  std::ofstream 
    outFile,
    outFile1;
  //
  outFile.open("Data_mesh.vs.conductivity.frame");
  outFile1.open("Centroid_normal.frame");
  //
  outFile << err.rdbuf();
  outFile1 << err1.rdbuf();
  //
  outFile.close();  
  outFile1.close();  
#endif
#endif      
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
