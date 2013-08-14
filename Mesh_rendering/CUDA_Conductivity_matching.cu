//
// Project
//
#include "CUDA_Conductivity_matching.h"
//
//
//
__global__ void 
process_kernel()
{
};

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
								 float* Voxel_center_pos_z
								 ):
  size_of_array_( Size_of_array )
{
  //
  // Memory allocation on CUDA device
  cudaMalloc( (void**)&positions_array_x_, 
	      Size_of_array * sizeof(float));
  cudaMalloc( (void**)&positions_array_y_, 
	      Size_of_array * sizeof(float));
  cudaMalloc( (void**)&positions_array_z_, 
	      Size_of_array * sizeof(float));

  //
  // Copy the array on GPU
  cudaMemcpy( positions_array_x_, Voxel_center_pos_x, Size_of_array * sizeof(float),
	      cudaMemcpyHostToDevice);
  cudaMemcpy( positions_array_y_, Voxel_center_pos_y, Size_of_array * sizeof(float),
	      cudaMemcpyHostToDevice);
  cudaMemcpy( positions_array_z_, Voxel_center_pos_z, Size_of_array * sizeof(float),
	      cudaMemcpyHostToDevice);
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
int* 
Domains::CUDA_Conductivity_matching::find_vertices_voxel_index( float* Vertices_position )
{
  //
  //
  int* vertices_voxel_index = new int[5];
  
  //
  // Cuda kernel
  // size_of_array_  = 100 * 100 * 60
  // size_of_array_ /= 64 = 9375
  process_kernel <<< size_of_array_ / 64, 64 >>> ();

  //
  //
  return vertices_voxel_index;
}
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
