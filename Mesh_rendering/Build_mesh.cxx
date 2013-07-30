#include <stdio.h>
#include <sstream>
#include <math.h>       /* round, floor, ceil, trunc */
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
// We give a comprehensive type name
//
typedef Domains::Build_mesh Domains_build_mesh;
typedef Domains::Access_parameters DAp;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;
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
  Eigen::Matrix< float, 3, 3 > rotation = (DAp::get_instance())->get_rotation_();
//  rotation << 
//    -1, 2.98023e-08, -6.14673e-08,
//    -5.58793e-09, 3.14321e-08, 1, 
//    3.72529e-08, -1, 4.22005e-08;
  //
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
//  translation <<
//    128.007,
//    -89.8845,
//    133.433;

  //
  // typedef
  typedef typename C3t3::Triangulation Triangulation;
  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
  typedef typename Triangulation::Vertex_handle Vertex_handle;
  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
  typedef typename Triangulation::Point Point_3;
  //
  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
// typedef typename No_patch_facet_pmap_second<C3t3,Cell_pmap> Facet_pmap_twice;
  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;

  //
  // Output FEniCS xml file
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
//  rotation << 
//    -1, 2.98023e-08, -6.14673e-08,
//    -5.58793e-09, 3.14321e-08, 1, 
//    3.72529e-08, -1, 4.22005e-08;
  //
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
//  translation <<
//    128.007,
//    -89.8845,
//    133.433;

  //
  // typedef
  typedef typename C3t3::Triangulation Triangulation;
  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
  typedef typename Triangulation::Vertex_handle Vertex_handle;
  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
  typedef typename Triangulation::Point Point_3;
  //
  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
// typedef typename No_patch_facet_pmap_second<C3t3,Cell_pmap> Facet_pmap_twice;
  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;

  //
  // Output FEniCS xml file
  std::ofstream VTU_xml_file("mesh.vtu");
  
  //
  // Vertices
  const Triangulation& triangulation = mesh_.triangulation();
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
  // typedef
  typedef typename C3t3::Triangulation Triangulation;
  typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
  typedef typename Triangulation::Vertex_handle Vertex_handle;
  typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
  typedef typename Triangulation::Point Point_3;
  //
  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;

  //
  // Retrieve the transformation matrix and vector from aseg and conductivity
  Eigen::Matrix< float, 3, 3 > rotation                = (DAp::get_instance())->get_rotation_();
  Eigen::Matrix< float, 3, 1 > translation             = (DAp::get_instance())->get_translation_();
  Eigen::Matrix< float, 3, 3 > eigenvalues_rotation    = (DAp::get_instance())->get_eigenvalues_rotation_();
  Eigen::Matrix< float, 3, 1 > eigenvalues_translation = (DAp::get_instance())->get_eigenvalues_translation_();
  
  //
  // Retrieve voxelization information from conductivity
  int eigenvalues_number_of_pixels_x = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
  int eigenvalues_number_of_pixels_y = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
  int eigenvalues_number_of_pixels_z = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();

  //
  // Retrieve the conductivity data array to match the cell's mesh
  Eigen::Matrix <float, 3, 3>* conductivity_tensors_array = nullptr;
  Eigen::Matrix <float, 3, 1>* positions_array = nullptr;
  bool* Do_we_have_conductivity = nullptr; 
  //
  (DAp::get_instance())->get_conductivity_tensors_array_( &conductivity_tensors_array );
  (DAp::get_instance())->get_positions_array_( &positions_array );
  (DAp::get_instance())->get_Do_we_have_conductivity_( &Do_we_have_conductivity );
  
  //
  // Vertices
  const Triangulation& triangulation = mesh_.triangulation();
  Cell_pmap            cell_pmap( mesh_ );
  Facet_pmap           facet_pmap( mesh_, cell_pmap );
  Vertex_pmap          vertex_pmap( mesh_, cell_pmap, facet_pmap );
  //
  std::map<Vertex_handle, int> V;

  //
  // Main loop
  int 
    num_tetrahedra = 0;
  Point_3 
    CGAL_cell_vertices[4],
    CGAL_cell_centroid;
//  Eigen::Matrix< float, 3, 1 > cell_vertices[4];
  Eigen::Matrix< float, 3, 1 > cell_centroid;
  //
  Eigen::Matrix< float, 3, 3 > 
    tensor_v0,
    tensor_v1,
    tensor_v2,
    tensor_v3,
    tensor_c;
  int count[10] = {0,0,0,0,0,0,0,0,0,0};
#ifdef TRACE
#if ( TRACE == 100 )
  std::stringstream err;
  err << "X_centroid Y_centroid Z_centroid  l1  l2  l3 l_para l_perpend l_mean \n";
#endif
#endif

  //
  for( Cell_iterator cit = mesh_.cells_in_complex_begin() ;
       cit != mesh_.cells_in_complex_end() ;
       ++cit )
    {
      //
      // Vertices positions and centroid of the cell
      CGAL_cell_vertices[0] = cit->vertex( 0 )->point();
      CGAL_cell_vertices[1] = cit->vertex( 1 )->point();
      CGAL_cell_vertices[2] = cit->vertex( 2 )->point();
      CGAL_cell_vertices[3] = cit->vertex( 3 )->point();
      // centroid
      CGAL_cell_centroid = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);
      cell_centroid <<
	(float)CGAL_cell_centroid.x(),
	(float)CGAL_cell_centroid.y(),
	(float)CGAL_cell_centroid.z();
      // move points from data to framework
      cell_centroid = rotation * cell_centroid + translation;
//      for (int i = 0 ; i < 4 ; i++)
//	cell_vertices[i] = rotation * cell_vertices[i] + translation;

      //
      // 
      if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION    &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SCALP      &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SKULL      &&
	  cell_pmap.subdomain_index( cit ) != CEREBROSPINAL_FLUID )
	{
	  //
	  //
	  int 
	    index_val = 0,
	    index_min_distance = 0;
	  float 
	    distance = 0.,
	    distance_min = 100000000.;
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
		      // find the position minimizing the distance with the centroid
		      distance = 
			/*sqrt(*/ 
			(positions_array[ index_val ](0) - cell_centroid(0)) * 
			(positions_array[ index_val ](0) - cell_centroid(0)) +
			(positions_array[ index_val ](1) - cell_centroid(1)) * 
			(positions_array[ index_val ](1) - cell_centroid(1)) +
			(positions_array[ index_val ](2) - cell_centroid(2)) * 
			(positions_array[ index_val ](2) - cell_centroid(2)) /*)*/;
		      //
		      if ( distance < distance_min )
			{
			  distance_min = distance;
			  index_min_distance = index_val;
			}
		    }
		}
	  //
	  tensor_c = conductivity_tensors_array[ index_min_distance ];
#ifdef TRACE
#if ( TRACE == 100 )
	  if ( cell_pmap.subdomain_index( cit ) != WHITE_MATTER )
	    {
	      Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float, 3, 3> > eigensolver( tensor_c );
	      if (eigensolver.info() != Eigen::Success) 
		abort();
	      //
	      float 
		a = eigensolver.eigenvalues()(0),
		b = eigensolver.eigenvalues()(1),
		c = eigensolver.eigenvalues()(2);
	      count[1] += c;
	      count[2] += (a+b)/2.;
	      err 
		<< cell_centroid(0) << " " << cell_centroid(1) << " " << cell_centroid(2) << " " 
		<< c << " " << b << " " << a << " " 
		<< c << " " << (a+b)/2. << " " << (a+b+c)/3. << std::endl;
	    }
#endif
#endif
	}
      //
      //
      num_tetrahedra++;
    }
  std::cout << "tetrahedra: " << num_tetrahedra << std::endl;;
  std::cout << "0 tensors: " << count[0] << std::endl;;
  std::cout << "1 tensors: " << count[1] << std::endl;;
  std::cout << "2 tensors: " << count[2] << std::endl;;
  std::cout << "3 tensors: " << count[3] << std::endl;;
  std::cout << "4 tensors: " << count[4] << std::endl;;
  std::cout << "5 tensors: " << count[5] << std::endl;;

  //
  // Clean up
  delete [] conductivity_tensors_array;
  conductivity_tensors_array = nullptr;
  delete [] positions_array;
  positions_array = nullptr;
  delete [] Do_we_have_conductivity; 
  Do_we_have_conductivity = nullptr; 

  //
  // 
#ifdef TRACE
#if ( TRACE == 100 )
  std::ofstream outFile;
  outFile.open("Data_mesh.vs.conductivity_All_Gray_Matter.frame");
  outFile << err.rdbuf();
  outFile.close();  
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
