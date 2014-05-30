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
#include <stdio.h>
#include <cstdio>
#include <sstream>
#include <algorithm> // std::for_each()
//
// UCSF
//
#include "Spheres_conductivity_tensor.h"
#include "Access_parameters.h"
//
// VTK
//
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>
// Transformation
#include <vtkTransform.h>
// Geometry
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkArrowSource.h>
#include <vtkSphereSource.h>
#include <vtkCubeSource.h>
#include <vtkConeSource.h>
#include <vtkAxesActor.h>
#include <vtkGlyph3D.h>
// Rendering
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
//#include <vtkProgrammableGlyphFilter.h>
// Include the mesh data
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkDataSetMapper.h>
//
// We give a comprehensive type name
//
typedef Domains::Spheres_conductivity_tensor DSct;
typedef Domains::Access_parameters DAp;
//
//
//
DSct::Spheres_conductivity_tensor()
{}
//
//
//
DSct::Spheres_conductivity_tensor( const DSct& that )
{
//  std::copy( &that.get_tab(),  &that.get_tab() + 4, tab_ );
////  tab_[0] = ( &that.get_tab() )[0];
////  tab_[1] = ( &that.get_tab() )[1];
////  tab_[2] = ( &that.get_tab() )[2];
////  tab_[3] = ( &that.get_tab() )[3];
}
//
//
//
DSct::Spheres_conductivity_tensor( DSct&& that )
{
//  // pilfer the source
//  list_position_ = std::move( that.list_position_ );
//  pos_x_ =  that.get_pos_x();
//  pos_y_ =  that.get_pos_y();
//  tab_   = &that.get_tab();
//  // reset that
//  that.set_pos_x( 0 );
//  that.set_pos_y( 0 );
//  that.set_tab( nullptr );
}
//
//
//
DSct& 
DSct::operator = ( const DSct& that )
{
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
  return *this;
}
//
//
//
DSct& 
DSct::operator = ( DSct&& that )
{
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
  //
  return *this;
}
//
//
//
void 
DSct::make_conductivity( const C3t3& Mesh )
{
  //
  // Tetrahedra mapping
  Cell_pmap cell_pmap( Mesh );

  //
  // Retrieve the transformation matrix and vector from aseg
  Eigen::Matrix< float, 3, 3 > rotation    = (DAp::get_instance())->get_rotation_();
  Eigen::Matrix< float, 3, 1 > translation = (DAp::get_instance())->get_translation_();
  
//  //
//  // Retrieve voxelization information from conductivity
//  int eigenvalues_number_of_pixels_x = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
//  int eigenvalues_number_of_pixels_y = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
//  int eigenvalues_number_of_pixels_z = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();
//
//  //
//  // Build the CGAL k-nearest neighbor tree
//  Tree tree_conductivity_positions;
//  int 
//    index_val = 0;
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
//	  if( Do_we_have_conductivity_[ index_val ] )
//	    tree_conductivity_positions.insert( std::make_tuple( Point_3( positions_array_[ index_val ](0),
//									  positions_array_[ index_val ](1),
//									  positions_array_[ index_val ](2) ), 
//								 index_val) );
//	}
  
  //
  // Main loop
  Point_3 
    CGAL_cell_vertices[5],
    CGAL_cell_centroid;
  Eigen::Matrix< float, 3, 1 > cell_vertices[5];
  //
  int inum = 0; 
  for( Cell_iterator cit = Mesh.cells_in_complex_begin() ;
       cit != Mesh.cells_in_complex_end() ;
       ++cit )
    {
      //
      // 
      int cell_id        = inum++;
      int cell_subdomain = cell_pmap.subdomain_index( cit );

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


      ////////////////////////
      // Brain segmentation //
      ////////////////////////
      if( cell_pmap.subdomain_index( cit ) != NO_SEGMENTATION     &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SCALP       &&
	  cell_pmap.subdomain_index( cit ) != OUTSIDE_SKULL       &&
	  cell_pmap.subdomain_index( cit ) != CEREBROSPINAL_FLUID &&
	  cell_pmap.subdomain_index( cit ) != ELECTRODE )
	{
//	  //
//	  // Search the K-nearest neighbor
//	  K_neighbor_search search( tree_conductivity_positions, 
//				    Point_3( cell_vertices[4/* centroid */](0),
//					     cell_vertices[4/* centroid */](1),
//					     cell_vertices[4/* centroid */](2) ), 
//				    /* K = */ 15);
//	  // Get the iterator on the nearest neighbor
//	  auto conductivity_centroids = search.begin();
//
//	  //
//	  // Select the conductivity cell with positive l3
//	  while( conductivity_centroids != search.end() &&
//		 eigen_values_matrices_array_[std::get<1>( conductivity_centroids->first )](2,2) < 0. )
//	    conductivity_centroids++;
//	  //
//	  if( conductivity_centroids == search.end() )
//	    {
//	      std::cerr << "You might think about increasing the number of neighbor. Or check the Diffusion/Conductivity file." << std::endl;
//	      exit(1);
//	    }
//
//	  //
//	  // create the cell conductivity information object
//	  Eigen::Vector3f eigen_vector[3];
//	  for ( int i = 0 ; i < 3 ; i++ )
//	    {
//	      eigen_vector[i] <<
//		P_matrices_array_[std::get<1>( conductivity_centroids->first )](0,i),
//		P_matrices_array_[std::get<1>( conductivity_centroids->first )](1,i),
//		P_matrices_array_[std::get<1>( conductivity_centroids->first )](2,i);
//	      //
//	      Eigen::Vector3f eigen_vector_tmp = rotation * eigen_vector[i];
//	      eigen_vector[i] = eigen_vector_tmp;
//	    }
	  //
	  
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.33,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.33,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.33,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.33, /*C00*/
			      0.00, /*C01*/
			      0.00, /*C02*/
			      0.33, /*C11*/
			      0.00, /*C12*/
			      0.33  /*C22*/ );


	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // end of brain //

      /////////////////////////
      // CerebroSpinal Fluid //
      /////////////////////////
      else if ( cell_pmap.subdomain_index( cit ) == CEREBROSPINAL_FLUID )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      1.79,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      1.79,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      1.79,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      1.79, /*C00*/
			      0.00, /*C01*/
			      0.00, /*C02*/
			      1.79, /*C11*/
			      0.00, /*C12*/
			      1.79  /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // end of CSF //

      ///////////
      // Skull //
      ///////////
      else if ( cell_pmap.subdomain_index( cit ) == OUTSIDE_SKULL )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.0132,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.0132,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.0132,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.0132, /*C00*/
			      0.00,   /*C01*/
			      0.00,   /*C02*/
			      0.0132, /*C11*/
			      0.00,   /*C12*/
			      0.0132  /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // and of scalp and skull //

      ///////////
      // Scalp //
      ///////////
      else if ( cell_pmap.subdomain_index( cit ) == OUTSIDE_SCALP )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.33,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.33,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.33,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.33, /*C00*/
			      0.00, /*C01*/
			      0.00, /*C02*/
			      0.33, /*C11*/
			      0.00, /*C12*/
			      0.33  /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // and of scalp  

      ///////////////
      // Electrode //
      ///////////////
      else if ( cell_pmap.subdomain_index( cit ) == ELECTRODE )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.33,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.33,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.33,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.33, /*C00*/
			      0.00, /*C01*/
			      0.00, /*C02*/
			      0.33, /*C11*/
			      0.00, /*C12*/
			      0.33  /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // Electrode
      else
	{
	  // Error condition
	  //
	  //
//	  Cell_conductivity 
//	    cell_parameters ( cell_id, cell_subdomain,
//			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
//			      0.,/* l1 */
//			      0., 0., 0., /* eigenvec V1 */
//			      0.,/* l2 */
//			      0., 0., 0., /* eigenvec V2 */
//			      0.,/* l3 */
//			      0., 0., 0., /* eigenvec V3 */
//			      0., /*C00*/
//			      0., /*C01*/
//			      0., /*C02*/
//			      0., /*C11*/
//			      0., /*C12*/
//			      0.  /*C22*/ );
//	  
//	  //
//	  // Add link to the list
//	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	}
    }// end of for( Cell_iterator cit = mesh_...


  //
  // Output for R analysis
  Make_analysis();
}
//
//
//
void
DSct::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
  //
  //
  output_stream_
    << "Cell_sub_domain "
    << "X_cent Y_cent Z_cent  "
    << "l1  l2  l3 l_long l_tang l_mean "
    << "v11 v12 v13 "
    << "v21 v22 v23 "
    << "v31 v32 v33 \n";


  //
  // Main loop
  for( auto cell_it : list_cell_conductivity_ )
    {
      output_stream_
	<< cell_it.get_cell_subdomain_() << " "
	<< (cell_it.get_centroid_lambda_()[0]).x() << " " 
	<< (cell_it.get_centroid_lambda_()[0]).y() << " " 
	<< (cell_it.get_centroid_lambda_()[0]).z() << " ";
      //
      float
	l1 = (cell_it.get_centroid_lambda_()[0]).weight(),
	l2 = (cell_it.get_centroid_lambda_()[1]).weight(),
	l3 = (cell_it.get_centroid_lambda_()[2]).weight();
      //
      output_stream_
	<< l1 << " " << l2 << " " << l3 << " " << l1 << " " 
	<< (l2+l3)/2. << " " << (l1+l2+l3)/3. << " " ;
      //
      output_stream_
	<< (cell_it.get_centroid_lambda_()[0]).vx() << " " 
	<< (cell_it.get_centroid_lambda_()[0]).vy() << " " 
	<< (cell_it.get_centroid_lambda_()[0]).vz() << " "
	<< (cell_it.get_centroid_lambda_()[1]).vx() << " " 
	<< (cell_it.get_centroid_lambda_()[1]).vy() << " " 
	<< (cell_it.get_centroid_lambda_()[1]).vz() << " "
	<< (cell_it.get_centroid_lambda_()[2]).vx() << " " 
	<< (cell_it.get_centroid_lambda_()[2]).vy() << " " 
	<< (cell_it.get_centroid_lambda_()[2]).vz() << " ";
      //
      output_stream_ << std::endl;
    }


  //
  // 
  Make_output_file("Data_mesh.vs.conductivity.frame");
#endif
#endif      
}
//
//
//
void 
DSct::Output_mesh_conductivity_xml()
{
  //
  // Output FEniCS conductivity xml files. 
  // We fillup the triangular sup from the symetric conductivity tensor
  std::string
    C00_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_(),
    C01_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_(),
    C02_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_(),
    C11_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_(),
    C12_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_(),
    C22_XML = (Domains::Access_parameters::get_instance())->get_files_path_output_();
  //
  C00_XML += std::string("C00.xml");
  C01_XML += std::string("C01.xml");
  C02_XML += std::string("C02.xml");
  C11_XML += std::string("C11.xml");
  C12_XML += std::string("C12.xml");
  C22_XML += std::string("C22.xml");
  //
  std::ofstream 
    FEniCS_xml_C00(C00_XML.c_str()), FEniCS_xml_C01(C01_XML.c_str()), FEniCS_xml_C02(C02_XML.c_str()), 
    FEniCS_xml_C11(C11_XML.c_str()), FEniCS_xml_C12(C12_XML.c_str()), 
    FEniCS_xml_C22(C22_XML.c_str());
  //
  int num_of_tetrahedra = list_cell_conductivity_.size();
  

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
  for ( auto cell_it : list_cell_conductivity_ )
    {
      //
      // C00
      FEniCS_xml_C00 
	<< "      <value cell_index=\"" <<  cell_it.get_cell_id_()
	<< "\" local_entity=\"0\" value=\""
	<< cell_it.C00()
	<< "\" />\n";
 
      //
      // C01
      FEniCS_xml_C01 
	<< "      <value cell_index=\"" <<  cell_it.get_cell_id_()
	<< "\" local_entity=\"0\" value=\""
	<< cell_it.C01()
	<< "\" />\n";
 
      //
      // C02
      FEniCS_xml_C02 
	<< "      <value cell_index=\"" <<  cell_it.get_cell_id_()
	<< "\" local_entity=\"0\" value=\""
	<< cell_it.C02()
	<< "\" />\n";
 
      //
      // C11
      FEniCS_xml_C11 
	<< "      <value cell_index=\"" <<  cell_it.get_cell_id_()
	<< "\" local_entity=\"0\" value=\""
	<< cell_it.C11()
	<< "\" />\n";
 
      //
      // C12
      FEniCS_xml_C12 
	<< "      <value cell_index=\"" <<  cell_it.get_cell_id_()
	<< "\" local_entity=\"0\" value=\""
	<< cell_it.C12()
	<< "\" />\n";
 
      //
      // C22
      FEniCS_xml_C22 
	<< "      <value cell_index=\"" <<  cell_it.get_cell_id_()
	<< "\" local_entity=\"0\" value=\""
	<< cell_it.C22()
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
DSct::VTK_visualization()
{}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DSct& that)
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
