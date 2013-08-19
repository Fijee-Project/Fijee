//
// Project
//
#include "CUDA_Conductivity_matching.h"
#include "CUDA_Conductivity_matching_functions.h"
//
//
//
Domains::CUDA_Conductivity_matching::CUDA_Conductivity_matching():
  positions_array_x_(NULL), positions_array_y_(NULL), positions_array_z_(NULL)
{
}
//
//
//
Domains::CUDA_Conductivity_matching::CUDA_Conductivity_matching( int Size_of_array,
								 float* Voxel_center_pos_x,
								 float* Voxel_center_pos_y,
								 float* Voxel_center_pos_z,
								 bool*  Do_we_have_conductivity ):
  size_of_array_( Size_of_array )
{

  //
  //
  cudaError_t err;
  
  //
  // Memory allocation on CUDA device
  err = cudaMalloc( (void**)&positions_array_x_, 
		    Size_of_array * sizeof(float));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMalloc( (void**)&positions_array_y_, 
		    Size_of_array * sizeof(float));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMalloc( (void**)&positions_array_z_, 
		    Size_of_array * sizeof(float));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMalloc( (void**)&do_we_have_conductivity_, 
		    Size_of_array * sizeof(bool));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }


  //
  // Copy the array on GPU
  err = cudaMemcpy( positions_array_x_, Voxel_center_pos_x, Size_of_array * sizeof(float),
		    cudaMemcpyHostToDevice);
  if( err != cudaSuccess )
    {
      printf( "CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMemcpy( positions_array_y_, Voxel_center_pos_y, Size_of_array * sizeof(float),
		    cudaMemcpyHostToDevice);
  if( err != cudaSuccess )
    {
      printf( "CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMemcpy( positions_array_z_, Voxel_center_pos_z, Size_of_array * sizeof(float),
		    cudaMemcpyHostToDevice);
  if( err != cudaSuccess )
    {
      printf( "CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMemcpy( do_we_have_conductivity_, Do_we_have_conductivity, Size_of_array * sizeof(bool),
		    cudaMemcpyHostToDevice);
  if( err != cudaSuccess )
    {
      printf( "CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }
}
// //
// //
// //
// Domains::CUDA_Conductivity_matching( const Domains& that )
// {
// }
// //
// //
// //
// Domains::CUDA_Conductivity_matching( Domains&& that )
// {
// }
//
//
//
Domains::CUDA_Conductivity_matching::~CUDA_Conductivity_matching()
{
  cudaFree( positions_array_x_ );
  cudaFree( positions_array_y_ );
  cudaFree( positions_array_z_ );
}
// //
// //
// //
// Domains::CUDA_Conductivity_matching& 
// Domains::CUDA_Conductivity_matching::operator = ( const Domains& that )
// {
//   if ( this != &that ) 
//     {
// //      // free existing ressources
// //      if( tab_ )
// //	{
// //	  delete [] tab_;
// //	  tab_ = nullptr;
// //	}
// //      // allocating new ressources
// //      pos_x_ = that.get_pos_x();
// //      pos_y_ = that.get_pos_y();
// //      list_position_ = that.get_list_position();
// //      //
// //      tab_ = new int[4];
// //      std::copy( &that.get_tab(),  &that.get_tab() + 4, tab_ );
//     }
//   //
//   return *this;
// }
// //
// //
// //
// Domains::CUDA_Conductivity_matching& 
// Domains::CUDA_Conductivity_matching::operator = ( Domains&& that )
// {
//   if( this != &that )
//     {
// //      // initialisation
// //      pos_x_ = 0;
// //      pos_y_ = 0;
// //      delete [] tab_;
// //      tab_   = nullptr;
// //      // pilfer the source
// //      list_position_ = std::move( that.list_position_ );
// //      pos_x_ =  that.get_pos_x();
// //      pos_y_ =  that.get_pos_y();
// //      tab_   = &that.get_tab();
// //      // reset that
// //      that.set_pos_x( 0 );
// //      that.set_pos_y( 0 );
// //      that.set_tab( nullptr );
//     }
//   //
//   return *this;
// }
//
//
//
void
Domains::CUDA_Conductivity_matching::find_vertices_voxel_index( float* Vertices_position,
								float* Point_min_distance,
								int*   Point_min_distance_index)
{
  //
  //
  cudaError_t err;
  float *cell_points;
  float *point_min_distance;
  int   *point_min_distance_index;
  
  //
  // Memory allocation on CUDA device
  err = cudaMalloc( (void**)&cell_points, 
		    5 * 3 * sizeof(float));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMalloc( (void**)&point_min_distance, 
		    (BLOCKS+REMAIN) * 5 * sizeof(float));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMalloc( (void**)&point_min_distance_index, 
		   (BLOCKS+REMAIN) * 5 * sizeof(int));
  if( err != cudaSuccess )
    {
      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
      abort();
    }

  
  //
  // Copy the array on GPU
  err = cudaMemcpy( cell_points, Vertices_position, 
		    5 * 3 * sizeof(float),
		    cudaMemcpyHostToDevice );
  if( err != cudaSuccess )
    {
      printf( "1 - CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }


  //
  // Cuda kernel
  // size_of_array_  = 100 * 100 * 60
  // size_of_array_ /= 64 = 9375
  process_kernel <<< BLOCKS, THREADS >>> ( cell_points, /* input  */
					   positions_array_x_,       /* already on GPU */
					   positions_array_y_,       /* already on GPU */
					   positions_array_z_,       /* already on GPU */
					   do_we_have_conductivity_, /* already on GPU */
					   point_min_distance, /* output */
					   point_min_distance_index);/* output */


  //
  // Copy the array results from GPU to host
  err = cudaMemcpy( Point_min_distance, point_min_distance, 
		    (BLOCKS+REMAIN) * 5 * sizeof(float),
		    cudaMemcpyDeviceToHost );
  if( err != cudaSuccess )
    {
      printf( "2 - CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }
  //
  err = cudaMemcpy( Point_min_distance_index, point_min_distance_index, 
		    (BLOCKS+REMAIN) * 5 * sizeof(int),
		    cudaMemcpyDeviceToHost );
  if( err != cudaSuccess )
    {
      printf( "3 - CUDA copy failled: %s", cudaGetErrorString(err) );
      abort();
    }


  //
  //
  cudaFree(cell_points);
  cudaFree(point_min_distance);
  cudaFree(point_min_distance_index);
}
////
////
////
//void
//Domains::CUDA_Conductivity_matching::find_vertices_voxel_index( float* Vertices_position,
//								float* Point_min_distance,
//								int* Point_min_distance_index)
//{
//  //
//  //
//  cudaError_t err;
//  float *cell_points;
//  float *point_min_distance;
//  int   *point_min_distance_index;
//  
//  //
//  // Memory allocation on CUDA device
//  err = cudaMalloc( (void**)&cell_points, 
//		    5 * 3 * sizeof(float));
//  if( err != cudaSuccess )
//    {
//      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
//      abort();
//    }
//  //
//  err = cudaMalloc( (void**)&point_min_distance, 
//		    BLOCKS * 5 * sizeof(float));
//  if( err != cudaSuccess )
//    {
//      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
//      abort();
//    }
//  //
//  err = cudaMalloc( (void**)&point_min_distance_index, 
//		    BLOCKS * 5 * sizeof(int));
//  if( err != cudaSuccess )
//    {
//      printf( "CUDA memory allocation failled: %s", cudaGetErrorString(err) );
//      abort();
//    }
//
//  
//  //
//  // Copy the array on GPU
//  err = cudaMemcpy( cell_points, Vertices_position, 
//		    15 * sizeof(float),
//		    cudaMemcpyHostToDevice );
//  if( err != cudaSuccess )
//    {
//      printf( "1CUDA copy failled: %s", cudaGetErrorString(err) );
//      abort();
//    }
//
//
//  //
//  // Cuda kernel
//  // size_of_array_  = 100 * 100 * 60
//  // size_of_array_ /= 64 = 9375
//  process_kernel <<< BLOCKS, THREADS >>> ( cell_points, /* input  */
//					   positions_array_x_, /* already on GPU */
//					   positions_array_y_, /* already on GPU */
//					   positions_array_z_, /* already on GPU */
//					   point_min_distance, /* output */
//					   point_min_distance_index);/* output */
//
//
//  //
//  // Copy the array results from GPU to host
//  err = cudaMemcpy( Point_min_distance, point_min_distance, 
//		    BLOCKS * 5 * sizeof(float),
//		    cudaMemcpyDeviceToHost );
//  if( err != cudaSuccess )
//    {
//      printf( "2CUDA copy failled: %s", cudaGetErrorString(err) );
//      abort();
//    }
//  //
//  err = cudaMemcpy( Point_min_distance_index, point_min_distance_index, 
//		    BLOCKS * 5 * sizeof(int),
//		    cudaMemcpyDeviceToHost );
//  if( err != cudaSuccess )
//    {
//      printf( "3CUDA copy failled: %s", cudaGetErrorString(err) );
//      abort();
//    }
//
//
//  //
//  //
//  cudaFree(cell_points);
//  cudaFree(point_min_distance);
//  cudaFree(point_min_distance_index);
//}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
			const Domains::CUDA_Conductivity_matching& that)
{
//  std::for_each( that.get_list_position().begin(),
//		 that.get_list_position().end(),
//		 [&stream]( int Val )
//		 {
//		   stream << "list pos = " << Val << "\n";
//		 });
//  //
//  stream << "positions minimum = " 
//	 << that.get_min_x() << " "
//	 << that.get_min_y() << " "
//	 << that.get_min_z() << "\n";
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
