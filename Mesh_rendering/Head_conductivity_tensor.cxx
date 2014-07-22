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
#include "Head_conductivity_tensor.h"
#include "Access_parameters.h"
// 
// Scotch
// 
extern "C"
{
#include <stdint.h>
#include <stdlib.h>     /* calloc, exit, free */
  // 
  // Scotch
  // 
//#include <scotch.h>
//#include <ptscotch.h>
  // 
  // Metis
  // 
}
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
// Conversion diffusion to conductivity
// Diffusion dimension
// [D] = m^{2}.s^{-1}
// Diffusion dimension in articles
// [D] = 10^{-3} x mm^{2}.s^{-1} = 10^{-9} x m^{2}.s^{-1}
//
// "Conductivity tensor mapping of the human brain using diffusion tensor MRI" 
// (David S. Tuch et al)
// sig_{nu} = k(d_{nu} -  d_{eps})
// 
// k   = 0.844 pm 0.0545 S.s / mm^{3}
// [k] = 10^{9} S.s.m^{-3}
// d_{eps}   = 0.124 pm 0.0540 µm^{2} / ms
// [d_{eps}] = 10^{-9} m^{2}.s^{-1}
//
// [sig_{nu}] = S/m
//
#define K_MAPPING 0.844 // 10^{9} S.s . m^{-3}
#define D_EPS     0.124 // 10^{-9} m^{2} . s^{-1}
//
// We give a comprehensive type name
//
typedef Domains::Head_conductivity_tensor DHct;
typedef Domains::Access_parameters DAp;
//
//
//
DHct::Head_conductivity_tensor()
{
  //
  // Header image's information MUST be the same for eigen values et vectores
  number_of_pixels_x_ = (DAp::get_instance())->get_eigenvalues_number_of_pixels_x_();
  number_of_pixels_y_ = (DAp::get_instance())->get_eigenvalues_number_of_pixels_y_();
  number_of_pixels_z_ = (DAp::get_instance())->get_eigenvalues_number_of_pixels_z_();
  //
  size_of_pixel_size_x_ = (DAp::get_instance())->get_eigenvalues_size_of_pixel_size_x_();
  size_of_pixel_size_y_ = (DAp::get_instance())->get_eigenvalues_size_of_pixel_size_y_();
  size_of_pixel_size_z_ = (DAp::get_instance())->get_eigenvalues_size_of_pixel_size_z_();

  //
  // Rotation Matrix and translation vector
  // The rotation matrix is based on the METHOD 3 of the nifti file (nifti1.h)
  rotation_    = (DAp::get_instance())->get_eigenvalues_rotation_();
  //
  translation_ = (DAp::get_instance())->get_eigenvalues_translation_();

  //
  // Allocate buffer and read first 4D volume from data file
  float* data_eigen_values  = nullptr;
  float* data_eigen_vector1 = nullptr;
  float* data_eigen_vector2 = nullptr;
  float* data_eigen_vector3 = nullptr;
  //
  (DAp::get_instance())->get_data_eigen_values_ ( &data_eigen_values  );
  (DAp::get_instance())->get_data_eigen_vector1_( &data_eigen_vector1 );
  (DAp::get_instance())->get_data_eigen_vector2_( &data_eigen_vector2 );
  (DAp::get_instance())->get_data_eigen_vector3_( &data_eigen_vector3 );
  //
  if ( data_eigen_values == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer data_eigen_values \n");
    exit(1);
  }
  if ( data_eigen_vector1 == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer data_eigen_vector1 \n");
    exit(1);
  }
  if ( data_eigen_vector2 == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer data_eigen_vector2 \n");
    exit(1);
  }
  if ( data_eigen_vector3 == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer data_eigen_vector3 \n");
    exit(1);
  }
  
  //
  // 
  float 
    l1 = 0., // eigen value 1
    l2 = 0., // eigen value 2
    l3 = 0., // eigen value 3
    //
    v1x = 0., /* eigen vector 1*/ v2x = 0., /* eigen vector 2 */ v3x = 0., /* eigen vector 3 */
    v1y = 0., /* eigen vector 1*/ v2y = 0., /* eigen vector 2 */ v3y = 0., /* eigen vector 3 */
    v1z = 0., /* eigen vector 1*/ v2z = 0., /* eigen vector 2 */ v3z = 0.; /* eigen vector 3 */
  int 
    index_val  = 0, 
    index_val1 = 0, 
    index_val2 = 0, 
    index_val3 = 0, 
    index_coeff = 0;
  //
  Eigen::Matrix < float, 3, 1 > index_mapping;
  //
  Eigen::Matrix < float, 3, 1 > position;
  Eigen::Matrix < float, 3, 1 > move_to_cell_center;
  move_to_cell_center <<
    size_of_pixel_size_x_ / 2.,
    size_of_pixel_size_y_ / 2.,
    size_of_pixel_size_z_ / 2.;

  //
  // initialization
  eigen_values_matrices_array_ = new Eigen::Matrix < float, 3, 3 > [ number_of_pixels_x_ 
								     * number_of_pixels_y_ 
								     * number_of_pixels_z_ ];
  P_matrices_array_            = new Eigen::Matrix < float, 3, 3 > [ number_of_pixels_x_ 
								     * number_of_pixels_y_ 
								     * number_of_pixels_z_ ];
  conductivity_tensors_array_  = new Eigen::Matrix < float, 3, 3 > [ number_of_pixels_x_ 
								     * number_of_pixels_y_ 
								     * number_of_pixels_z_ ];
  positions_array_             = new Eigen::Matrix < float, 3, 1 > [ number_of_pixels_x_ 
								     * number_of_pixels_y_ 
								     * number_of_pixels_z_ ];
  Do_we_have_conductivity_     = new bool [ number_of_pixels_x_ 
					    * number_of_pixels_y_ 
					    * number_of_pixels_z_ ];
  
  //
  // Output for R analysis
#ifdef TRACE
#if ( TRACE == 100 )
  std::stringstream err;
  err << "l1 true \t l2 true \t l3 true \t l1 \t l2 \t l3 \t delta12 \t delta13 \t delta23 \n";
#endif
#endif
  //
  for ( int dim3 = 0 ; dim3 < number_of_pixels_z_ ; dim3++)
    for ( int dim2 = 0 ; dim2 < number_of_pixels_y_ ; dim2++)
      for ( int dim1 = 0 ; dim1 < number_of_pixels_x_ ; dim1++)
	{
	  //
	  // Eigen values and vectors from nifti file
	  // All the frames are transfered in aseg referential
	  //

	  //
	  // Select the index
	  index_coeff = dim1 * 9
	    + dim2 * 9 * number_of_pixels_x_ 
	    + dim3 * 9 * number_of_pixels_x_ * number_of_pixels_y_;
	  // Select the index
	  index_val = index_val1 = dim1 
	    + dim2 * number_of_pixels_x_ 
	    + dim3 * number_of_pixels_x_ * number_of_pixels_y_;
	  //
	  index_val2 = dim1 
	    + dim2 * number_of_pixels_x_ 
	    + dim3 * number_of_pixels_x_ * number_of_pixels_y_ 
	    + 1 * number_of_pixels_x_ * number_of_pixels_y_ * number_of_pixels_z_;
	  //
	  index_val3 = dim1 
	    + dim2 * number_of_pixels_x_ 
	    + dim3 * number_of_pixels_x_ * number_of_pixels_y_ 
	    + 2 * number_of_pixels_x_ * number_of_pixels_y_ * number_of_pixels_z_;

	  //
	  // Diffusion eigenvalues -> conductivity eigenvalues
	  // sig_{nu} = k*(d_{nu} - d_{epsi})
	  //
	  
	  //
	  // eigen value 1: l1
	  l1 = 1000 * data_eigen_values[ index_val1 ];
	  l1 = K_MAPPING * (l1 - D_EPS);
	  // eigen value 2: l2
	  l2 = 1000 * data_eigen_values[ index_val2 ];
	  l2 = K_MAPPING * (l2 - D_EPS);
	  // eigen value 3: l3
	  l3 = 1000 * data_eigen_values[ index_val3 ];
	  l3 = K_MAPPING * (l3 - D_EPS);

	  //
	  // Diffusion eigenvalues == conductivity eigenvalues
	  // eigen vector 1:
	  v1x = data_eigen_vector1[ index_val1 ];
	  v1y = data_eigen_vector1[ index_val2 ];
	  v1z = data_eigen_vector1[ index_val3 ];

	  //
	  // eigen vector 2:
	  v2x = data_eigen_vector2[ index_val1 ];
	  v2y = data_eigen_vector2[ index_val2 ];
	  v2z = data_eigen_vector2[ index_val3 ];

	  //
	  // eigen vector 3:
	  v3x = data_eigen_vector3[ index_val1 ];
	  v3y = data_eigen_vector3[ index_val2 ];
	  v3z = data_eigen_vector3[ index_val3 ];
	  
	  //
	  // Fill up arrays
	  //
	  
	  //
	  // Position of the eigenvalues in the orig/aseg framework
	  position  << dim1, dim2, dim3;
	  position += move_to_cell_center;
	  // nifti data to study frame work
	  position = rotation_ * position + translation_;
	  // Mesh rendering framework (aseg/orig framework)
	  positions_array_[ index_val ] = position;
	  
	  //
	  // Eigen values matrices
	  Eigen::Matrix <float, 3, 3> D;
	  D << 
	    l1, 0.f, 0.f,
	    0.f, l2, 0.f,
	    0.f, 0.f, l3;
	  //
	  eigen_values_matrices_array_[ index_val ] = D;

	  //
	  //  Change of basis matrices
	  Eigen::Matrix <float, 3, 3> P;
	  P << 
	    v1x, v2x, v3x,
	    v1y, v2y, v3y,
	    v1z, v2z, v3z;
	  //
	  P_matrices_array_[ index_val ] = P;
	 
	  //
	  //  Conductivity tensor
	  //
	  Eigen::Matrix< float, 3, 3> conductivity_tensor;
	  //
	  if ( P.determinant() != 0.f )
	    {
	      conductivity_tensor = 
		rotation_ * (P * D * P.inverse()) * rotation_.inverse();
	      Do_we_have_conductivity_[ index_val ] = true;
	    }
	  else
	    {
	      conductivity_tensor << 
		0.f,0.f,0.f,
		0.f,0.f,0.f,
		0.f,0.f,0.f;
	      //
	      Do_we_have_conductivity_[index_val] = false;
	    }
	  //
	  conductivity_tensors_array_[ index_val ] = conductivity_tensor;

#ifdef TRACE
#if ( TRACE == 2 )
	  // 
	  // Print some results
	  //
	  std::cout << std::endl;
	  std::cout << "dim1 = x = " << dim1 
		    << " ; dim2 = y = " << dim2 << " (" << dim2*number_of_pixels_x_ 
		    << ") ; dim3 = z = " << dim3 << " (" << dim3*number_of_pixels_x_*number_of_pixels_y_
		    << ")" << std::endl;
	  std::cout << std::endl;

	  // 
	  // Eigen values matrices array
	  std::cout << "Eigen values: " << endl;
	  std::cout << eigen_values_matrices_array_[ index_val ] 
		    << std::endl;

	  //
	  // Change of basis matrices
	  std::cout << std::endl;
	  std::cout << "Change of basis matrices: " << endl;
	  std::cout << P_matrices_array_[ index_val ] 
		    << std::endl;

	  //
	  // Algebra propreties of the transformation
	  std::cout << std::endl;
	  std::cout << "Transformation matrix S " << endl;
	  std::cout << rotation_ << std::endl;
	  std::cout << "Determinant matrix S " << endl;
	  std::cout << rotation_.determinant() << std::endl;
	  std::cout << "S^-1 = S^T " << endl;
	  std::cout << rotation_.inverse() << std::endl;
	  std::cout << rotation_.transpose() << std::endl;

	  //
	  // Conductivity tensors array
	  std::cout << std::endl;
	  std::cout << "Conductivity tensors array: " << endl;
	  std::cout << conductivity_tensors_array_[ index_val ] 
		    << std::endl;
	  //
	  // Check the solution
	  Eigen::Matrix<float, 3, 3> tensor = 
	    conductivity_tensors_array_[ index_val ];
	  //
	  if ( P.determinant() != 0.f )
	    {
	      Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float, 3, 3> > eigensolver( tensor );
	      if (eigensolver.info() != Eigen::Success) 
		abort();
	      //
	      std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
	      std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
			<< "corresponding to these eigenvalues:\n"
			<< eigensolver.eigenvectors() << std::endl;
	    }
#endif
#endif

	  //
	  // Output for R analysis
#ifdef TRACE
#if ( TRACE == 100 )
	  //
	  // Check the solution
	  Eigen::Matrix<float, 3, 3> tensor = 
	    conductivity_tensors_array_[ index_val ];
	  //
	  if ( P.determinant() != 0.f )
	    {
	      Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float, 3, 3> > eigensolver( tensor );
	      if (eigensolver.info() != Eigen::Success) 
		abort();
	      err 
		<< eigen_values_matrices_array_[ index_val ](0,0) << " "
		<< eigen_values_matrices_array_[ index_val ](1,1) << " "
		<< eigen_values_matrices_array_[ index_val ](2,2) << " ";
	      err << eigensolver.eigenvalues()(2) << " " << eigensolver.eigenvalues()(1) << " " << eigensolver.eigenvalues()(0) << " ";
	      err 
		<< conductivity_tensors_array_[ index_val ](0,1) - conductivity_tensors_array_[ index_val ](1,0) << " " 
		<< conductivity_tensors_array_[ index_val ](0,2) - conductivity_tensors_array_[ index_val ](2,0) << " " 
		<< conductivity_tensors_array_[ index_val ](1,2) - conductivity_tensors_array_[ index_val ](2,1) << std::endl;
	      //
	    }
#endif
#endif
#ifdef TRACE
#if ( TRACE == 3 )
	  //
	  // Recreate the ascii output from dt_recon -> 
	  if( dim1 == 0 )
	    std::cout << std::endl;
	  //
	  if( dim2 == 0 && dim1 == 0 )
	    std::cout << std::endl;
	  //
	  if ( P.determinant() != 0.f )
	    std::cout << eigen_values_matrices_array_[ index_val ] (0,0) << " ";
	  else
	    std::cout << "0 " ;
#endif
#endif
	}

  //
  // Clean up memory
  delete [] data_eigen_values;
  data_eigen_values = nullptr;
  delete [] data_eigen_vector1;
  data_eigen_vector1 = nullptr;
  delete [] data_eigen_vector2;
  data_eigen_vector2 = nullptr;
  delete [] data_eigen_vector3;
  data_eigen_vector3 = nullptr;

  //
  // Output for R analysis 
#ifdef TRACE
#if ( TRACE == 100 )
  std::ofstream outFile;
  outFile.open("Data.frame");
  outFile << err.rdbuf();
  outFile.close();  
#endif
#endif
}
////
////
////
//DHct::Head_conductivity_tensor( const DHct& that ):
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
////
////
////
//DHct::Head_conductivity_tensor( DHct&& that ):
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
DHct::~Head_conductivity_tensor()
{
  // Eigen values matrices array
  delete [] eigen_values_matrices_array_;
  eigen_values_matrices_array_ = nullptr;
  // Change of basis matrices array
  if ( P_matrices_array_ != nullptr )
    {
      delete [] P_matrices_array_;
      P_matrices_array_ = nullptr;
    }
  // Conductivity tensors array
  if ( conductivity_tensors_array_ != nullptr )
    {
      delete [] conductivity_tensors_array_;
      conductivity_tensors_array_ = nullptr;
    }
  // positions array
  if ( positions_array_ != nullptr )
    {
      delete [] positions_array_;
      positions_array_ = nullptr;
    }
  // 
  if ( Do_we_have_conductivity_ != nullptr )
    {
      delete [] Do_we_have_conductivity_;
      Do_we_have_conductivity_ = nullptr;
    }
}
////
////
////
//DHct& 
//DHct::operator = ( const DHct& that )
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
////
////
////
//DHct& 
//DHct::operator = ( DHct&& that )
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
DHct::make_conductivity( const C3t3& Mesh )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Head_conductivity_tensor::make_conductivity");

  //
  // Tetrahedra mapping
  Cell_pmap cell_pmap( Mesh );

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
	  if( Do_we_have_conductivity_[ index_val ] )
	    tree_conductivity_positions.insert( std::make_tuple( Point_3( positions_array_[ index_val ](0),
									  positions_array_[ index_val ](1),
									  positions_array_[ index_val ](2) ), 
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
	  cell_pmap.subdomain_index( cit ) != AIR_IN_SKULL        &&
	  cell_pmap.subdomain_index( cit ) != SPONGIOSA_SKULL     &&
	  cell_pmap.subdomain_index( cit ) != EYE                 &&
	  cell_pmap.subdomain_index( cit ) != ELECTRODE )
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
	  // Select the conductivity cell with positive l3
	  while( conductivity_centroids != search.end() &&
		 eigen_values_matrices_array_[std::get<1>( conductivity_centroids->first )](2,2) < 0. )
	    conductivity_centroids++;
	  //
	  if( conductivity_centroids == search.end() )
	    {
	      std::cerr << "You might think about increasing the number of neighbor. Or check the Diffusion/Conductivity file." << std::endl;
	      exit(1);
	    }

	  //
	  // create the cell conductivity information object
	  Eigen::Vector3f eigen_vector[3];
	  for ( int i = 0 ; i < 3 ; i++ )
	    {
	      eigen_vector[i] <<
		P_matrices_array_[std::get<1>( conductivity_centroids->first )](0,i),
		P_matrices_array_[std::get<1>( conductivity_centroids->first )](1,i),
		P_matrices_array_[std::get<1>( conductivity_centroids->first )](2,i);
	      //
	      Eigen::Vector3f eigen_vector_tmp = rotation * eigen_vector[i];
	      eigen_vector[i] = eigen_vector_tmp;
	    }
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain, 0,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      eigen_values_matrices_array_[std::get<1>( conductivity_centroids->first )](0,0),/* l1 */
			      eigen_vector[0](0), eigen_vector[0](1), eigen_vector[0](2), /* eigenvec V1 */
			      eigen_values_matrices_array_[std::get<1>( conductivity_centroids->first )](1,1),/* l2 */
			      eigen_vector[1](0), eigen_vector[1](1), eigen_vector[1](2), /* eigenvec V2 */
			      eigen_values_matrices_array_[std::get<1>( conductivity_centroids->first )](2,2),/* l3 */
			      eigen_vector[2](0), eigen_vector[2](1), eigen_vector[2](2), /* eigenvec V3 */
			      conductivity_tensors_array_[std::get<1>( conductivity_centroids->first )](0,0), /*C00*/
			      conductivity_tensors_array_[std::get<1>( conductivity_centroids->first )](0,1), /*C01*/
			      conductivity_tensors_array_[std::get<1>( conductivity_centroids->first )](0,2), /*C02*/
			      conductivity_tensors_array_[std::get<1>( conductivity_centroids->first )](1,1), /*C11*/
			      conductivity_tensors_array_[std::get<1>( conductivity_centroids->first )](1,2), /*C12*/
			      conductivity_tensors_array_[std::get<1>( conductivity_centroids->first )](2,2)  /*C22*/ );
	  
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
	    cell_parameters ( cell_id, cell_subdomain, 0,
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
	    cell_parameters ( cell_id, cell_subdomain, 0,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.00552,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.00552,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.00552,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.00552, /*C00*/
			      0.00,   /*C01*/
			      0.00,   /*C02*/
			      0.00552, /*C11*/
			      0.00,   /*C12*/
			      0.00552  /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // and of scalp and skull //

      /////////////////////
      // Skull spongiosa //
      /////////////////////
      else if ( cell_pmap.subdomain_index( cit ) == SPONGIOSA_SKULL )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain, 0,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.01457,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.01457,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.01457,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.01457, /*C00*/
			      0.00,   /*C01*/
			      0.00,   /*C02*/
			      0.01457, /*C11*/
			      0.00,   /*C12*/
			      0.01457  /*C22*/ );
	  
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
	    cell_parameters ( cell_id, cell_subdomain, 0,
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

      //////////////////////
      // Air in the skull //
      //////////////////////
      else if ( cell_pmap.subdomain_index( cit ) == AIR_IN_SKULL )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain, 0,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      2.5e-14, /* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      2.5e-14, /* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      2.5e-14, /* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      2.5e-14,/*C00*/
			      0.00,   /*C01*/
			      0.00,   /*C02*/
			      2.5e-14,/*C11*/
			      0.00,   /*C12*/
			      2.5e-14 /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // Electrode

      /////////
      // EYE //
      /////////
      else if ( cell_pmap.subdomain_index( cit ) == EYE )
	{
	  //
	  // http://onlinelibrary.wiley.com/doi/10.1002/cnm.2483/pdf
	  // resistance 300 Ohm.cm -> 0.33 S/m -> 0.25 S/m
	  // Wolter 1.5 S/m
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain, 0,
			      cell_vertices[4](0),cell_vertices[4](1),cell_vertices[4](2),/* centroid */
			      0.25,/* l1 */
			      1., 0., 0., /* eigenvec V1 */
			      0.25,/* l2 */
			      0., 1., 0., /* eigenvec V2 */
			      0.25,/* l3 */
			      0., 0., 1., /* eigenvec V3 */
			      0.25,  /*C00*/
			      0.00, /*C01*/
			      0.00, /*C02*/
			      0.25,  /*C11*/
			      0.00, /*C12*/
			      0.25   /*C22*/ );
	  
	  //
	  // Add link to the list
	  list_cell_conductivity_.push_back( std::move(cell_parameters) );
	} // Electrode

      ///////////////
      // Electrode //
      ///////////////
      else if ( cell_pmap.subdomain_index( cit ) == ELECTRODE )
	{
	  //
	  //
	  Cell_conductivity 
	    cell_parameters ( cell_id, cell_subdomain, 0,
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
//	    cell_parameters ( cell_id, cell_subdomain, 0,
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
DHct::make_parcellation( const C3t3& Mesh )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Head_conductivity_tensor::make_parcellation");

  printf(" size of idx_t: %zubits, real_t: %zubits, idx_t *: %zubits\n", 
	 8*sizeof(idx_t), 8*sizeof(real_t), 8*sizeof(idx_t *));

  //
  // Tetrahedra mapping
  Cell_pmap cell_pmap( Mesh );
  // 
  int inum = 0; 
  int vertex_new_id = 0; 
  // list of elements (cells - tetrahedrons) and their nodes (tetrahedron's vertices)
  // just store addresses
  std::vector< Cell_iterator > elements_nodes;
  // 
  // 
  std::map< Tr::Vertex_handle, 
	    std::tuple<int/*vertex new id*/, std::list<int/*cell id*/> > > edge_vertex_to_element;
  // 
  int cell_id = 0;
  for( Cell_iterator cit = Mesh.cells_in_complex_begin() ;
       cit != Mesh.cells_in_complex_end() ; cit++ )
    {
      //
      if( cell_pmap.subdomain_index( cit ) == LEFT_GRAY_MATTER )
	{
	  elements_nodes.push_back( cit );
	  int mesh_cell = cell_id++;
	
	  // 
	  // 
	  for( int i = 0 ; i < 4 ; i++ )
	    {
	      // 
	      // 
	      std::map< Tr::Vertex_handle, std::tuple<int, std::list<int> > >::iterator it_v_handler;
	      // 
	      it_v_handler = edge_vertex_to_element.find(cit->vertex(i));
	      // The vertex is not in the edge_vertex_to_element map
	      if ( it_v_handler == edge_vertex_to_element.end() )
		{
		  edge_vertex_to_element.insert( std::pair<Tr::Vertex_handle,
						 std::tuple<int, std::list<int> > >
						 (cit->vertex(i),
						  std::make_tuple (vertex_new_id++,
								   std::list<int>(1, mesh_cell))) );
		}
	      // The vertex is in the edge_vertex_to_element map
	      else
		{
		  (std::get<1/*list of cell id*/>(it_v_handler->second)).push_back(mesh_cell);
		}
	    }
	}
    }
	

  // 
  // Metis data structure
  // 
  
  // 
  // 
  idx_t options[METIS_NOPTIONS];
  // 
  //  Options --------------------------------------------------------
  // ptype=kway, objtype=cut, ctype=shem, rtype=greedy, iptype=metisrb
  // dbglvl=0, ufactor=1.030, minconn=NO, contig=NO, nooutput=NO
  // seed=-1, niter=10, ncuts=1
  // gtype=dual, ncommon=1, niter=10, ncuts=1

  // 
  METIS_SetDefaultOptions(options);
  // METIS_PTYPE_RB || METIS_PTYPE_KWAY
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
  // Specifies the type of objective. Possible values are:
  // - METIS_OBJTYPE_CUT Edge-cut minimization.
  // - METIS_OBJTYPE_VOL Total communication volume minimization.
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  // Specifies the matching scheme to be used during coarsening. Possible values are:
  // - METIS_CTYPE_RM Random matching.
  // - METIS_CTYPE_SHEM Sorted heavy-edge matching.
  options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
//  // Determines the algorithm used during initial partitioning. Possible values are:
//  // - METIS_IPTYPE_GROW Grows a bisection using a greedy strategy.
//  // - METIS_IPTYPE_RANDOM Computes a bisection at random followed by a refinement.
//  // - METIS_IPTYPE_EDGE Derives a separator from an edge cut.
//  // - METIS_IPTYPE_NODE Grow a bisection using a greedy node-based strategy.
//  options[METIS_OPTION_IPTYPE]  = METIS_IPTYPE_GROW;
  // Determines the algorithm used for refinement. Possible values are:
  // - METIS_RTYPE_FM FM-based cut refinement.
  // - METIS_RTYPE_GREEDY Greedy-based cut and volume refinement.
  // - METIS_RTYPE_SEP2SIDED Two-sided node FM refinement.
  // - METIS_RTYPE_SEP1SIDED One-sided node FM refinement.
  options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
  // Debug
  options[METIS_OPTION_DBGLVL] = 0;
  //
  //  options[METIS_OPTION_UFACTOR] = params->ufactor;
  // 0 || 1
  options[METIS_OPTION_MINCONN] = 0;
  // 0 || 1
  options[METIS_OPTION_CONTIG] = 0;
  // Specifies the seed for the random number generator.
  options[METIS_OPTION_SEED] = -1;
  // Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process. Default is 10.
  options[METIS_OPTION_NITER] = 10;
  // Specifies the number of different partitionings that it will compute. The final partitioning 
  // is the one that achieves the best edgecut or communication volume. Default is 1.
  options[METIS_OPTION_NCUTS] = 1;
  //
  //  options[METIS_OPTION_NUMBERING] = 0;


  // 
  // number of elements and vertices
  idx_t 
    elem_nbr     = static_cast<idx_t>(elements_nodes.size()),
    vertices_nbr = static_cast<idx_t>(edge_vertex_to_element.size());
  // The size of the eptr array is n + 1, where n is the number of elements in the mesh. 
  // The size of the eind array is of size equal to the sum of the number of nodes in all 
  // the elements of the mesh. The list of nodes belonging to the ith element of the mesh 
  // are stored in consecutive locations of eind starting at position eptr[i] up to 
  // (but not including) position eptr[i+1].
  idx_t *eptr, *epart, *eind, *npart;
  //
  eptr  = (idx_t*)calloc(sizeof(idx_t), elem_nbr+1);
  eind  = (idx_t*)calloc(sizeof(idx_t), 4 * elem_nbr);
  //partition vector for the elements of the mesh
  epart = (idx_t*)calloc(sizeof(idx_t), elem_nbr);
  // partition vector for the nodes of the mesh
  npart = (idx_t*)calloc(sizeof(idx_t), vertices_nbr);

  // 
  // Build Metis mesh data structure
  // 
  std::cout << "Build Metis mesh data structure" << std::endl;
     
  // 
  // Load elements in eptr (element ptr) and eind (element index) arrays
  for ( int elem = 0 ; elem < elem_nbr ; elem++ )
    {
      eptr[elem] = 4*elem;
      // 
      for ( int num = 0 ; num < 4 ; num++ )
	{
	  auto vertex = edge_vertex_to_element.find(elements_nodes[elem]->vertex(num));
	  if( vertex != edge_vertex_to_element.end() )
	    eind[4*elem+num] = std::get<0>(vertex->second);
	  else
	    {
	      std::cerr << "Parcelletion: all vertices must be found:" << std::endl;
	      std::cerr << vertex->first->point() << std::endl;
	      abort();
	    }
	}
    }
  // last element
  eptr[ elem_nbr ] = 4*elem_nbr;

  std::cout << "Check: eptr" << std::endl;
  for (int elem = 0 ; elem < elem_nbr + 1 ; elem++)
    std::cout << "eptr[" << elem << "] = " << eptr[elem] << std::endl;

  std::cout << "Check: cell - vertex, size: " << elements_nodes.size() << std::endl;
  int id = 0;
  for (auto elem : elements_nodes )
    {
      for ( int i = 0 ; i < 4 ; i++ )
	{
	  auto vertex = edge_vertex_to_element.find(elem->vertex(i));
	  std::cout << "cell id: " << id << " - vertex(" << i << "): " << std::get<0>(vertex->second) 
		    << " : " << vertex->first->point()
		    << std::endl;
	}
      // 
      id++;
    }

  std::cout << "Check: vertex - cell, size: " << edge_vertex_to_element.size() << std::endl;
  for (auto vertex : edge_vertex_to_element )
    {
      std::cout << "vertex id: " << std::get<0>(vertex.second) << " list size: " << std::get<1>(vertex.second).size() << std::endl;
      for ( auto cell : std::get<1>(vertex.second) )
	std::cout << cell << " ";
      std::cout << std::endl;
    }


   std::cout << "Check - eind: cell - vertex" << std::endl;
   std::cout << elem_nbr << std::endl;
   for (int elem = 0 ; elem < elem_nbr ; elem++ )
     {
       for ( int num = 0 ; num < 4 ; num++ )
	 std::cout << eind[4*elem+num]+1 << " ";
       std::cout << std::endl;
     }

   


  // 
  // Metis partitioning
  // 
  int status  = 0;
  idx_t 
    // 1 - 2 elem share at least 1 vertex; 
    // 2 - 2 elem share at least 1 edge; 
    // 3 - 2 elem share at least 1 facet (triangl); ... 
    ncommon = 1,  /* Higher it is faster it is */
    nparts  = 16; /*number of part*/
  idx_t objval;

  std::cout << "Metis partitioning" << std::endl;
  // 
  switch (METIS_GTYPE_DUAL/*params->gtype*/) 
    {
      //
    case METIS_GTYPE_DUAL:
      {
	status = METIS_PartMeshDual( &elem_nbr, &vertices_nbr, eptr, eind, NULL, NULL, 
				     &ncommon, &nparts, NULL, options, &objval, epart, npart );
	break;
      }
      // 
    case METIS_GTYPE_NODAL:
      {
	status = METIS_PartMeshNodal( &elem_nbr, &vertices_nbr, eptr, eind, NULL, NULL, 
				     &nparts, NULL, options, &objval, epart, npart );
	break;
      }
      // 
    default:
      {
	abort();
      }
    }
  // 
  switch ( status ) 
    {
    case METIS_ERROR_INPUT://Indicates an input error
      {
	std::cerr << "input error" << std::endl;
	abort();
      }
    case METIS_ERROR_MEMORY://Indicates that it could not allocate the required memory.
      {
	std::cerr << "could not allocate the required memory" << std::endl;
	abort();
      }
    case METIS_ERROR://Indicates some other type of error
      {
	std::cerr << "ERROR" << std::endl;
	abort();
      }
    }
  
  std::cout << "X Y Z PAR" << std::endl;
  for( int cell = 0 ; cell < elem_nbr ; cell++ )
    {
      auto centroid = elements_nodes[cell];
      Point_3 
	CGAL_cell_vertices[4],
	CGAL_cell_centroid;
      // 
      for (int i = 0 ; i < 4 ; i++)
	  CGAL_cell_vertices[i] = centroid->vertex( i )->point();
      // 
      CGAL_cell_centroid = CGAL::centroid(CGAL_cell_vertices, CGAL_cell_vertices + 4);
      // 
      std::cout << CGAL_cell_centroid.point() << " " << epart[cell] << std::endl;
    }

}
//
//
//
void 
DHct::make_parcellation1( const C3t3& Mesh )
{
//  // 
//  // Time log
//  FIJEE_TIME_PROFILER("Domain::Head_conductivity_tensor::make_parcellation");
//
//  //
//  // Tetrahedra mapping
//  Cell_pmap cell_pmap( Mesh );
//  // 
//  int inum = 0; 
//  int vertex_new_id = 0; 
//  // list of elements (cells - tetrahedrons) and their nodes (tetrahedron's vertices)
//  // just store addresses
//  std::vector< Cell_iterator > elements_nodes;
//  // 
//  std::map< Tr::Vertex_handle, 
//	    std::tuple<int/*vertex new id*/, std::list<int/*cell id*/> > > edge_vertex_to_element;
//  // 
//  for( Cell_iterator cit = Mesh.cells_in_complex_begin() ;
//       cit != Mesh.cells_in_complex_end() ; cit++ )
//    if( cell_pmap.subdomain_index( cit ) == LEFT_GRAY_MATTER )
//      {
//	// 
//	// cell and index
//	int cell_id = inum++;
//	// increment the vector of elements_nodes corresponding to cell_id
//	elements_nodes.push_back( cit );
//	
//	// 
//	// 
//	for( int i = 0 ; i < 4 ; i++ )
//	  {
//	    // 
//	    // 
//	    std::map< Tr::Vertex_handle, std::tuple<int, std::list<int> > >::iterator it_v_handler;
//	    // 
//	    it_v_handler = edge_vertex_to_element.find(cit->vertex(i));
//	    // The vertex is not in the edge_vertex_to_element map
//	    if ( it_v_handler == edge_vertex_to_element.end() )
//	      {
//		edge_vertex_to_element.insert( std::pair<Tr::Vertex_handle,
//					       std::tuple<int, std::list<int> > >
//					       (cit->vertex(i),
//						std::make_tuple (vertex_new_id++,
//								 std::list<int>(1, cell_id))) );
//	      }
//	    // The vertex is in the edge_vertex_to_element map
//	    else
//	      {
//		(std::get<1/*list of cell id*/>(it_v_handler->second)).push_back(cell_id);
//	      }
//	  }
//      }
//     std::cout << "everything is good" << std::endl;
//  // 
//  // Build Scotch graph (Scotch and libScotch 6.0 User's Guide)
//  //
//
//  // Initialization
//  // Base value for element indexings.
//  SCOTCH_Num velmbas = 0;
//  // Base value for node indexings. 
//  // The base value of the underlying graph, baseval, is set as min(velmbas, vnodbas).
//  SCOTCH_Num vnodbas = static_cast< SCOTCH_Num >( elements_nodes.size() );
//  // Number of element vertices in mesh.
//  SCOTCH_Num velmnbr = vnodbas; //static_cast< SCOTCH_Num >( elements_nodes.size() );
//  // Number of node vertices in mesh. 
//  // The overall number of vertices in the underlying graph, vertnbr, is set as velmnbr + vnodnbr.
//  SCOTCH_Num vnodnbr = static_cast< SCOTCH_Num >( edge_vertex_to_element.size() );
//  // Number of arcs in mesh. 
//  // Since edges are represented by both of their ends, the number of edge data in the mesh is 
//  // twice the number of edges.
//  SCOTCH_Num edgenbr = 2 * 4 /*vertices in cell*/ * velmnbr;
//  // Array of start indices in edgetab of vertex (that is, both elements and nodes) 
//  // adjacency sub-arrays.
//  // verttab[baseval /*0*/+ vertnbr] = (baseval + edgenbr) = edgenbr
//  SCOTCH_Num* verttab;
//  verttab = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), velmnbr + vnodnbr);
//  // Array of after-last indices in edgetab of vertex adjacency sub-arrays. 
//  // For any element or node vertex i, with baseval i < (baseval + vertnbr), 
//  // vendtab[i] − verttab[i] is the degree of vertex i, and the indices of the neighbors of i 
//  // are stored in edgetab from edgetab[verttab[i]] to edgetab [vendtab[i]−1], inclusive.
//  SCOTCH_Num* vendtab;
//  vendtab = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), velmnbr + vnodnbr);
//  // SCOTCH_Num* vendtab = NULL;
//  //  SCOTCH_Num vendtab[velmnbr + vnodnbr];
//  //  vendtab = (verttab + 1);
//
//  // 
//  // Bipartite element-node graph construction
//  // 
//
//
//  std::cout << "0 step: inum = " << inum << "then become 0" << std::endl;
//  std::cout << "vnodbas = " << vnodbas <<std::endl;
//  std::cout << "velmnbr = " << velmnbr <<std::endl;
//  std::cout << "vnodnbr = " << vnodnbr <<std::endl;
//  std::cout << "edgenbr = " << edgenbr <<std::endl;
//  
//  // 
//  // 
//  std::list< SCOTCH_Num > edgetab_list;
//  
//  // 
//  // First, we fill the elements
//  inum = 0; 
//  //
//  for ( Cell_iterator cit : elements_nodes )
//    {
//      // 
//      // 
//      int cell_id = inum++;
//      // Each element (cell) has 4 vertices
//      verttab[ cell_id ] = 4 * cell_id;
//      vendtab[ cell_id ] = 4 * cell_id + 4;
//      std::cout << "verttab[" <<  cell_id << " ] = " << verttab[ cell_id ] << std::endl;
//      std::cout << "vendtab[" <<  cell_id << " ] = " << vendtab[ cell_id ] << std::endl;
//      // 
//      // 
//      for ( int i = 0 ; i < 4 ; i++ )
//	{
//	  // 
//	  auto it_vertex = edge_vertex_to_element.find( cit->vertex(i) );
//	  // 
//	  if( it_vertex != edge_vertex_to_element.end() )
//	    {
//	      // 
//	      // we shift the vertex id after the element number (velmnbr)
//	      // edgetab[4*cell_id+i]=velmnbr+static_cast<SCOTCH_Num>(std::get<0>(it_vertex->second));
//	      edgetab_list.push_back(velmnbr+static_cast<SCOTCH_Num>(std::get<0>(it_vertex->second)));
//	    }
//	  else
//	    {
//	      std::cerr << "Parcelletion: all vertices must be found:" << std::endl;
//	      std::cerr << it_vertex->first->point() << std::endl;
//	      abort();
//	    }
//	}
//    }
//
//      std::cout << "First step: inum = " << inum <<std::endl;
//      std::cout << "vnodbas = " << vnodbas <<std::endl;
//      std::cout << "velmnbr = " << velmnbr <<std::endl;
//      std::cout << "vnodnbr = " << vnodnbr <<std::endl;
//      std::cout << "edgenbr = " << edgenbr <<std::endl;
//      std::cout << "edgenbr_list = 4*velmnbr = " << edgetab_list.size() <<std::endl;
//
//  
//  // 
//  // Second, we fill the nodes
//  int edgetab_vertex_pos = 4*inum;
//  std::cout << "vertex start pos = " <<  edgetab_vertex_pos<<std::endl;
//  for ( auto it_vertex : edge_vertex_to_element )
//    {
//      // 
//      // 
//      verttab[ inum ]     = edgetab_vertex_pos;
//      std::cout << "verttab[" <<  inum  << " ] = " << verttab[ inum ] << std::endl;
//      edgetab_vertex_pos += static_cast<SCOTCH_Num>( std::get<1>(it_vertex.second).size() );
//      vendtab[ inum++ ]   = edgetab_vertex_pos;
//      std::cout << "vendtab[" <<  inum - 1 << " ] = " << vendtab[ inum - 1 ] 
//		<< std::endl;
//      //
//      for ( auto cell_id : std::get<1>(it_vertex.second) )
//	{
//	  // edgetab_vertex_pos += element_connected++; 
//	  // edgetab[edgetab_vertex_pos] = static_cast<SCOTCH_Num>(cell_id);
//	  edgetab_list.push_back(static_cast<SCOTCH_Num>(cell_id));
//	}
//    }
//  // last element
//  //  verttab[ inum ] = edgetab_vertex_pos;
//
//      std::cout << "Second step: inum = " << inum <<std::endl;
//      std::cout << "vnodbas = " << vnodbas <<std::endl;
//      std::cout << "velmnbr = " << velmnbr <<std::endl;
//      std::cout << "vnodnbr = " << vnodnbr <<std::endl;
//      std::cout << "edgenbr = " << edgenbr <<std::endl;
//      std::cout << "edgenbr_list = " << edgetab_list.size() <<std::endl;
//
//      std::cout << "copy edgetab_list in edgetab" <<std::endl;
//  
//  // 
//  // edgetab is the adjacency array, of size at least edgenbr 
//  // (it can be more if the edge array is not compact).
//  SCOTCH_Num* edgetab;
//  edgetab = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), edgetab_list.size());
//  std::copy(edgetab_list.begin(), edgetab_list.end(), edgetab);
//
//  // 
//  if( vendtab[ --inum ] != edgenbr )
//    {
//      std::cerr << "vnodbas = " << vnodbas <<std::endl;
//      std::cerr << "velmnbr = " << velmnbr <<std::endl;
//      std::cerr << "vnodnbr = " << vnodnbr <<std::endl;
//      std::cerr << "edgenbr = " << edgenbr <<std::endl;
//      std::cerr << "verttab[vertnbr] = " << verttab[velmnbr+vnodnbr] <<std::endl;
//      std::cerr << "Parcelletion: we should have: verttab[vertnbr] = edgenbr" << std::endl;
//      abort();
//    }
//
//  // 
//  // Scotch Mesh handling
//  SCOTCH_Num* velotab = NULL;
//  SCOTCH_Num* vlbltab = NULL;
//  SCOTCH_Num* vnlotab = NULL;
//  // Graph construction
//  SCOTCH_Mesh* meshptr;
//  meshptr = SCOTCH_meshAlloc();
//  //  meshptr = ((SCOTCH_Mesh *) memAlloc (sizeof (SCOTCH_Mesh)));
//  std::cout << "Init the Scotch mesh" <<std::endl;
//  if ( SCOTCH_meshInit( meshptr ) != 0 )
//    {
//      std::cerr << "Parcelletion: SCOTCH_meshInit abort" << std::endl;
//      abort();
//    }
//  //
//  std::cout << "Build the Scotch mesh" <<std::endl;
//  if ( SCOTCH_meshBuild( meshptr, velmbas, vnodbas, velmnbr, vnodnbr,
//			 verttab, vendtab, velotab, vnlotab, vlbltab, 
//			 edgenbr, edgetab) != 0 )
//    {
//      std::cerr << "Parcelletion: SCOTCH_meshBuild abort" << std::endl;
//      abort();
//    }
//
//
//#ifdef TRACE
//  // 
//  // At least in the development phase, it is recommended to check the Scotch mesh
//  // SCOTCH_meshCheck
//  std::cout << "Check the Scotch mesh" <<std::endl;
////  if ( SCOTCH_meshCheck( meshptr ) != 0 )
////    {
////      std::cerr << "Parcelletion: SCOTCH_meshCheck abort" << std::endl;
////      abort();
////    }
//#endif      
//  
//  // 
//  // Graph construction
//  SCOTCH_Graph* grafptr;
//  grafptr = SCOTCH_graphAlloc();
//  SCOTCH_graphInit( grafptr );
//  // 
//  std::cout << "Build graph from the Scotch mesh" <<std::endl;
//  if ( SCOTCH_meshGraph( meshptr, grafptr ) != 0 )
//    {
//      std::cerr << "Parcelletion: SCOTCH_meshGraph abort" << std::endl;
//      abort();
//    }
//
//  //
//  // Partitioning strategy
//  SCOTCH_Strat* strat;
//  strat = SCOTCH_stratAlloc();
//  SCOTCH_stratInit(strat);
//  //
//  SCOTCH_Num* vertices_partition;
//  vertices_partition = (SCOTCH_Num*)calloc(sizeof(SCOTCH_Num), vnodnbr);
// 
//  // Partition graph
//  std::cout << "Graph partitioning" <<std::endl;
//  if (SCOTCH_graphPart(grafptr, 16, strat, vertices_partition))
//  {
//    std::cerr << "Parcelletion: SCOTCH_graphPart abort" << std::endl;
//    abort();
//  }
//  std::cout << "VERTEX LIST" <<std::endl;
//  std::cout << "X Y Z PAR" <<std::endl;
//  for ( auto vertex : edge_vertex_to_element )
//    std::cout << (vertex.first)->point() << " " 
//	      << vertices_partition[std::get<0>(vertex.second)]
//	      << std::endl;
//
//
//  std::cout << "Free Scotch objects" <<std::endl;
//  // 
//  // Free the structures
//  // Array and SCOTCH_meshExit
//  SCOTCH_meshExit(meshptr);
//  //
//  SCOTCH_graphExit(grafptr);
//  // 
//  SCOTCH_stratExit(strat);
//  //
//  free(verttab);
//  free(edgetab);
}
//
//
//
void
DHct::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
  //
  //
  output_stream_
    << "Cell_sub_domain Cell_parcel "
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
	<< cell_it.get_cell_parcel_() << " "
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
DHct::Output_mesh_conductivity_xml()
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
DHct::VTK_visualization()
{
  //
  // Create conductivity vector field
  //
   
  //
  //
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
 
  //
  //
  Eigen::Matrix< float, 3, 1 > position;
  Eigen::Matrix< float, 3, 1 > position_index;
  Eigen::Matrix< float, 3, 1 > main_direction;
  int index_val = 0;

  //
  // Anisotropy
  //
  
  //
  // Anisotropy direction
  vtkSmartPointer<vtkFloatArray> vectors = vtkSmartPointer<vtkFloatArray>::New();
  vectors->SetNumberOfComponents(3); // number of elements

  //
  // Intensity of anisotropy. Create the color map
  vtkSmartPointer<vtkLookupTable> colorLookupTable = 
    vtkSmartPointer<vtkLookupTable>::New();
  colorLookupTable->SetTableRange( 0.0 , 1.0 );
  colorLookupTable->Build();
  //
  vtkSmartPointer<vtkUnsignedCharArray> anisotropy = vtkSmartPointer<vtkUnsignedCharArray>::New();
  anisotropy->SetName("Anisotropy"); 
  anisotropy->SetNumberOfComponents(3); // number of elements
  unsigned char anisotropy_color[3];
  double        color[3];
  double        gradiant = 0;
  //
  for ( int dim3 = 0 ; dim3 < number_of_pixels_z_ ; dim3++ )
    for ( int dim2 = 0 ; dim2 < number_of_pixels_y_ ; dim2++) // Up
      for ( int dim1 = 0 ; dim1 < number_of_pixels_x_ ; dim1++ ) // Right
	{
	  index_val = dim1 + dim2 * number_of_pixels_x_ + dim3 * number_of_pixels_x_ * number_of_pixels_y_;
	  if( Do_we_have_conductivity_[ index_val ] )
	    {
//	      if( positions_array_[ index_val ](2,0) < 44 &&
//		  positions_array_[ index_val ](2,0) > 41 )
//		{
	      // position
	      points->InsertNextPoint(positions_array_[ index_val ](0,0), 
				      positions_array_[ index_val ](1,0), 
				      positions_array_[ index_val ](2,0));
//	      std::cout << positions_array_[ index_val ] << std::endl<< std::endl;
	      //
	      main_direction << 
		P_matrices_array_[index_val](0,0),
		P_matrices_array_[index_val](1,0),
		P_matrices_array_[index_val](2,0);
	      // Transformation with a proper/improper symtry matrix
	      main_direction = ( rotation_  / size_of_pixel_size_x_ ) * main_direction;
             // Mesh rendering framework
//	      main_direction = rotation_mesh_framework_ * main_direction;
	      //
	      float v[3] = {main_direction(0,0),
			    main_direction(1,0),
			    main_direction(2,0)};
	      vectors->InsertNextTupleValue( v );
	      // Anisotropy evaluation
	      gradiant  =      eigen_values_matrices_array_[index_val](1,1);
	      gradiant +=      eigen_values_matrices_array_[index_val](2,2); 
	      gradiant /=  2 * eigen_values_matrices_array_[index_val](0,0);
	      colorLookupTable->GetColor(gradiant, color);
	      anisotropy_color[0] = static_cast<unsigned char>( color[0] * 255. );
	      anisotropy_color[1] = static_cast<unsigned char>( color[1] * 255. );
	      anisotropy_color[2] = static_cast<unsigned char>( color[2] * 255. );
	      anisotropy->InsertNextTupleValue( anisotropy_color );
//		}
	    }
	}
  //
  vtkSmartPointer<vtkPolyData> polydata = 
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints( points );
  polydata->GetPointData()->SetVectors( vectors );
  polydata->GetPointData()->SetScalars( anisotropy );

  //
  // Create the glyphs
  //

  //
  // Create anything you want here
  vtkSmartPointer<vtkArrowSource> glyph_shape = vtkSmartPointer<vtkArrowSource>::New();
  glyph_shape->SetTipResolution( 6 );
  glyph_shape->SetTipRadius( 0.1 );
  glyph_shape->SetTipLength( 0.35 );
  glyph_shape->SetShaftResolution( 6 );
  glyph_shape->SetShaftRadius( 0.03 );

  //
  //
  vtkSmartPointer<vtkGlyph3D> glyph3D = 
    vtkSmartPointer<vtkGlyph3D>::New();
#if VTK_MAJOR_VERSION <= 5
  glyph3D->SetSource( glyph_shape->GetOutput() );
  glyph3D->SetInput( polydata );
#else
  glyph3D->SetSourceConnection( glyph_shape->GetOutputPort() );
  glyph3D->SetInputData( polydata );
#endif
  //
  glyph3D->SetVectorModeToUseVector();
  glyph3D->SetColorModeToColorByScalar();
  glyph3D->SetScaleModeToScaleByVector();
//  glyph3D->SetScaleModeToDataScalingOff();
  glyph3D->OrientOn();
  glyph3D->SetScaleFactor(1.5);
 //
  glyph3D->Update();
 
  //
  // Visualize
  //

  //
  // Mapper
  vtkSmartPointer<vtkPolyDataMapper> mapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(glyph3D->GetOutputPort());
 
  //
  // Actor
  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
 
  //
  // Render
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  //
  vtkSmartPointer<vtkRenderWindow> renderWindow = 
    vtkSmartPointer<vtkRenderWindow>::New();
  // an interactor
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  //
  // Axes
  vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
  // properties of the axes labels can be set as follows
  // this sets the x axis label to red
  // axes->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor(1,0,0);
  // the actual text of the axis label can be changed:
   axes->SetXAxisLabelText("");
   axes->SetYAxisLabelText("");
   axes->SetZAxisLabelText("");
   // translation of the axes
   Eigen::Matrix<float, 3, 1> translation;
   translation << 
     0.,
     0.,
     0.;
   //
   vtkSmartPointer<vtkTransform> transform =
     vtkSmartPointer<vtkTransform>::New();
   transform->Translate( translation(0,0), 
			 translation(1,0), 
			 translation(2,0) );
   // The axes are positioned with a user transform
   axes->SetUserTransform(transform);

  //
  // Add Actors to the scene
  renderer->AddActor( actor );
  renderer->AddActor( axes );
  // Background 
  renderer->SetBackground(.0, .0, .0); 
 
//  //
//  // Mesh import
//  //
// 
//  //
//  // Mesh as unscructured data
//  std::string filename = "mesh.vtu";
// 
//  //
//  //read all the data from the file
//  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
//    vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
//  reader->SetFileName( filename.c_str() );
//  reader->Update();
// 
//  //
//  //Create a mapper and actor
//  vtkSmartPointer<vtkDataSetMapper> mapper_unstructured_data =
//    vtkSmartPointer<vtkDataSetMapper>::New();
//  mapper_unstructured_data->SetInputConnection(reader->GetOutputPort());
//  //
//  vtkSmartPointer<vtkActor> actor_unstructured_data =
//    vtkSmartPointer<vtkActor>::New();
//  actor_unstructured_data->SetMapper(mapper_unstructured_data);
// 
//  //
//  //Add the actor to the scene
//  renderer->AddActor( actor_unstructured_data );
 
  //
  // Render and interact
  //

  renderWindow->Render();
  renderWindowInteractor->Start();
}
//
//
//
void 
DHct::INRIMAGE_image_of_conductivity_anisotropy()
{
  // 
  //Create the INRIMAGE
  //

  //
  // Data initialization
  std::ofstream* file_inrimage = new std::ofstream("conductivity.inr", std::ios::out | std::ios::binary);
  char* data_label    = new char[ 256 * 256 * 256 ];

  //
  // initialisation 256 * 256 * 256 voxels set at 0
  for ( int k = 0; k < 256; k++ )
    for ( int j = 0; j < 256; j++ )
      for ( int i = 0; i < 256; i++ )
	data_label[ i + j*256 + k*256*256 ] = 0;

  //
  //
  std::string header  = "#INRIMAGE-4#{\n";
  header +="XDIM=256\n"; // x dimension
  header +="YDIM=256\n"; // y dimension
  header +="ZDIM=256\n"; // z dimension
  header +="VDIM=1\n";   // number of scalar per voxel 
                         // (1 = scalar, 3 = 3D image of vectors)
  header +="TYPE=unsigned fixed\n";
  header +="PIXSIZE=8 bits\n"; // 8, 16, 32, or 64
  header +="CPU=decm\n";
  header +="VX=1\n"; // voxel size in x
  header +="VY=1\n"; // voxel size in y
  header +="VZ=1\n"; // voxel size in z
  std::string headerEND = "##}\n";
  
  //
  // output .inr header
  int hlen = 256 - header.length() - headerEND.length();
  for (int i = 0 ; i < hlen ; i++)
    header += '\n';
  //
  header += headerEND;
  file_inrimage->write(  header.c_str(), header.size() );

  //
  // Anisotropy
  //
  
  int    index_val = 0;
  int    index_mapping = 0;
  double gradiant  = 0;
  //
  for ( int dim3 = 0 ; dim3 < 256  ; dim3++ )
    for ( int dim2 = 0 ; dim2 < 256 ; dim2++ )
      for ( int dim1 = 0 ; dim1 < 256 ; dim1++ )
	{
	  index_val = dim1 + 256 * dim2 + 256 * 256 * dim3;
	  //	  index_mapping = nifti_data_to_diffusion_mapping_array_[ index_val ];
	  //
	  if ( index_mapping  != -1 )
	    {
//	    if( Do_we_have_conductivity_[ index_val ] )
//	      {
//	      position_index << 
//		size_of_pixel_size_x_ * dim1,
//		size_of_pixel_size_z_ * dim2,
//		qfac_* size_of_pixel_size_z_ * dim3;
//	      //
//	      position = rotation_ * position_index + translation_;
//	      //
//	      points->InsertNextPoint(position(0,0), position(1,0), position(2,0));
//	      vectors->InsertNextTuple3( P_matrices_array_[index_val](0,0),
//					 P_matrices_array_[index_val](1,0),
//					 P_matrices_array_[index_val](2,0));
//	      // Anisotropy evaluation
	    gradiant = ( eigen_values_matrices_array_[index_mapping](2,2) / 
			 eigen_values_matrices_array_[index_mapping](0,0) );
	  //
	  data_label[index_val] = static_cast<char>( gradiant * 100 );
//	      colorLookupTable->GetColor(gradiant, color);
//	      if ( gradiant > 0.9 && gradiant < 1.0 ){
//		std::cout << dim1 << " " << dim2 << " " << dim3 << std::endl;
//		std::cout << gradiant << std::endl;
//	      }
//	      anisotropy_color[0] = static_cast<unsigned char>( color[0] * 255. );
//	      anisotropy_color[1] = static_cast<unsigned char>( color[1] * 255. );
//	      anisotropy_color[2] = static_cast<unsigned char>( color[2] * 255. );
//	      anisotropy->InsertNextTupleValue( anisotropy_color );
	    }
	}
  
  //
  // write the inrimage file
  file_inrimage->write( data_label, 256*256*256 );
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DHct& that)
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
