#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include "CUDA_Conductivity_matching.h"
// 600 000      BLOCKS  THREADS REMAIN
//   9 375       9375     64     0
//   4 687.5	 4687    128    64
//   2 343.75	 2343    256   192
//   1 171.875	 1171    512   448  
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
// Threads = 512
//#define THREADS  512
//#define BLOCKS  1171
//#define REMAIN   448
//
#define VOXEL_LIMIT 3.65
// Number of tests
#define N 100

//using namespace std;

int
main(int argc, char* argv[])
{
  //
  // initialize random seed:
  srand (time(NULL));

  //
  //
  int array_size = 600000;
  float 
    voxel_center_position_x[array_size],
    voxel_center_position_y[array_size],
    voxel_center_position_z[array_size];
  bool Do_we_have_conductivity[array_size];
  //
  for (int i = 0 ; i  < array_size ; i++)
    {
      voxel_center_position_x[i] = (float)((rand() % 10000)/10000.f - 0.512f);
      voxel_center_position_y[i] = (float)((rand() % 10000)/10000.f - 0.598f);
      voxel_center_position_z[i] = (float)((rand() % 10000)/10000.f - 0.578f);
      int boolean = rand() % 10;
      Do_we_have_conductivity[i] = ( boolean > 7 ? true : false);
    }

  
  //
  // CUDA initialization
  Domains::CUDA_Conductivity_matching cuda_matcher( array_size,
						    voxel_center_position_x,
						    voxel_center_position_y,
						    voxel_center_position_z,
						    Do_we_have_conductivity );
  
  //
  //
  float distance_array[5*(BLOCKS+REMAIN)];
  int   distance_index_array[5*(BLOCKS+REMAIN)];
  //
  float cell_points[15];
  for(int i = 0 ; i < 15 ; i++)
    cell_points[i] = (float)((rand() % 100000)/1000 - 50.532f);


  //
  //
  float distance_min_gpu[5] = {1000000.f,1000000.f,1000000.f,1000000.f,1000000.f};
  int   distance_min_index_gpu[5] = {-1,-1,-1,-1,-1};
  //
  float distance[5] = {-1.f,-1.f,-1.f,-1.f,-1.f};
  float distance_min[5] = {1000000.f,1000000.f,1000000.f,1000000.f,1000000.f};
  int   distance_min_index[5] = {-1,-1,-1,-1,-1};
 
  //
  // Start time check
  auto start = std::chrono::monotonic_clock::now();
  int time = 0;
  while( ++time != N ){
    //
    // CUDA
    //
    if ( strcmp(argv[1], "gpu") == 0 || strcmp(argv[1], "both") == 0 ) 
      {
	// launch the kernel
	cuda_matcher.find_vertices_voxel_index( cell_points, 
						distance_array, 
						distance_index_array);
	// Post treatment of CUDA results
	for (int block = 0 ; block < (BLOCKS+REMAIN) ; block++ )
	  {
	    for ( int point = 0 ; point < 5 ; point++ )
	      {
		if( distance_array[5*block + point] < distance_min_gpu[point] )
		  {
		    distance_min_gpu[point] = distance_array[ 5*block + point ];
		    distance_min_index_gpu[point] = distance_index_array[5*block + point];
		  }
		//	  //
		//	  std::cout << distance_array[ 5*block + point ] << "   " 
		//		    << point << "   " << distance_index_array[5*block + point] 
		//		    << std::endl;
	      }
	  }
      }
    //
    // CPU
    //
    if ( strcmp(argv[1], "cpu") == 0 || strcmp(argv[1], "both") == 0 )
      {
	for ( int i = 0 ; i < array_size ; i++ )
	  {
	    if (Do_we_have_conductivity[i])
	      for ( int point = 0 ; point < 5 ; point++ )
		{
		  distance[point] = 
		    (voxel_center_position_x[i] - cell_points[3*point + 0])*(voxel_center_position_x[i] - cell_points[3*point + 0])+
		    (voxel_center_position_y[i] - cell_points[3*point + 1])*(voxel_center_position_y[i] - cell_points[3*point + 1])+
		    (voxel_center_position_z[i] - cell_points[3*point + 2])*(voxel_center_position_z[i] - cell_points[3*point + 2]);
	   
		  //
		  //	  std::cout << distance[point] << " " << point << " " << i << std::endl;
		  if( distance[point] < distance_min[point] && distance[point] > VOXEL_LIMIT )
		    {
		      distance_min[point] = distance[point];
		      distance_min_index[point] = i;
		    }
		}
	  }
      }

  }

  //
  // End of the time comparison
  auto end = std::chrono::monotonic_clock::now();
  auto duration = end - start;
  std::cout << std::chrono::duration <double, std::milli> (duration).count() << " ms" << std::endl;  
  
  //
  // Display the results
  for(int point = 0 ; point < 5 ; point++)
    std::cout << "CPU Point_v[" << point << "] = " << distance_min_index[point]  << " (" << distance_min[point] << ")" << std::endl;
  for(int point = 0 ; point < 5 ; point++)
    std::cout << "GPU Point_v[" << point << "] = " << distance_min_index_gpu[point]  << " (" << distance_min_gpu[point] << ")" << std::endl;

  //
  //
  return 1;
}


//
// TEST for 10 tetrahedra array
//int
//main()
//{
//  /* initialize random seed: */
//  srand (time(NULL));
//
//  int array_size = 600000;
//  float 
//    *voxel_center_position_x = new float[array_size],
//    *voxel_center_position_y = new float[array_size],
//    *voxel_center_position_z = new float[array_size];
//
//
//  for (int i = 0 ; i  < array_size ; i++)
//    {
//      voxel_center_position_x[i] = (float)((rand() % 10000)/10000.f - 0.512f);
//      voxel_center_position_y[i] = (float)((rand() % 10000)/10000.f - 0.598f);
//      voxel_center_position_z[i] = (float)((rand() % 10000)/10000.f - 0.578f);
//    }
//
//  Domains::CUDA_Conductivity_matching cuda_matcher( array_size,
//						    voxel_center_position_x,
//						    voxel_center_position_y,
//						    voxel_center_position_z );
//  
////  delete [] voxel_center_position_x;
////  voxel_center_position_x = nullptr;
////  delete [] voxel_center_position_y;
////  voxel_center_position_y = nullptr;
////  delete [] voxel_center_position_z;
////  voxel_center_position_z = nullptr;
//
//
//  float distance_array[10*5*BLOCKS];
//  int   distance_index_array[10*5*BLOCKS];
//
//  float cell_points[10*15];
//  float cell_points_seq[15];
//  //  for(int i = 0 ; i < 15 ; i++)
//  //    cell_points[i] = (float)((rand() % 100000)/1000 - 50.532f);
//
//  //
//  //
//  float distance_min_gpu[5] = {1000000.f,1000000.f,1000000.f,1000000.f,1000000.f};
//  int   distance_min_index_gpu[5] = {-1,-1,-1,-1,-1};
//  //
//  float distance[5] = {-1.f,-1.f,-1.f,-1.f,-1.f};
//  float distance_min[5] = {1000000.f,1000000.f,1000000.f,1000000.f,1000000.f};
//  int   distance_min_index[5] = {-1,-1,-1,-1,-1};
// 
//  ///////////////////////////
//  auto start = std::chrono::monotonic_clock::now();
//  int container = 0;
//  for ( int time = 0 ; time < 1000 ; time++, container++)
//    {
//      for(int i = 0 ; i < 15 ; i++)
//	cell_points_seq[i] = cell_points[15*container + i] = (float)((rand() % 100000)/1000 - 50.532f);
//
//     // CUDA
//       if ( container == 9 )
//	{
//	  cuda_matcher.find_vertices_voxel_index( cell_points, 
//						  distance_array, 
//						  distance_index_array);
//	  //
////	  for ( container = 0 ; container < 10 ; container++)
////	    for (int block = 0 ; block < BLOCKS ; block++ )
////	      {
////		for ( int point = 0 ; point < 5 ; point++ ){
////		  if( distance_array[5*BLOCKS*container + 5*block + point] < distance_min_gpu[point] )
////		    {
////		      distance_min_gpu[point] = distance_array[ 5*BLOCKS*container + 5*block + point ];
////		      distance_min_index_gpu[point] = distance_index_array[5*BLOCKS*container + 5*block + point];
////		    }
//////		  std::cout << distance_array[ 5*BLOCKS*container + 5*block + point ] << "   " << point << "   " << distance_index_array[5*BLOCKS*container + 5*block + point] << std::endl;
////		  }
////	      }
//	  //
//	  container = 0;
//	}
//
////// LINEAR
////      for ( int i = 0 ; i < array_size ; i++ )
////	{
////	  for ( int point = 0 ; point < 5 ; point++ )
////	    {
////	      distance[point] = 
////		(voxel_center_position_x[i] - cell_points_seq[3*point + 0])*(voxel_center_position_x[i] - cell_points_seq[3*point + 0])+
////		(voxel_center_position_y[i] - cell_points_seq[3*point + 1])*(voxel_center_position_y[i] - cell_points_seq[3*point + 1])+
////		(voxel_center_position_z[i] - cell_points_seq[3*point + 2])*(voxel_center_position_z[i] - cell_points_seq[3*point + 2]);
////	      
////	      //
////	      //	  std::cout << distance[point] << " " << point << " " << i << std::endl;
////	      if( distance[point] < distance_min[point] )
////		{
////		  distance_min[point] = distance[point];
////		  distance_min_index[point] = i;
////		}
////	    }
////	}
//
//    }
//  auto end = std::chrono::monotonic_clock::now();
//  auto duration = end - start;
//  std::cout << std::chrono::duration <double, std::milli> (duration).count() << " ms" << std::endl;  
//  ///////////////////////////
//
//
//  for(int point = 0 ; point < 5 ; point++)
//    std::cout << "Point_v[" << point << "] = " << distance_min_index[point]  << " (" << distance_min[point] << ")" << std::endl;
//  for(int point = 0 ; point < 5 ; point++)
//    std::cout << "GPU Point_v[" << point << "] = " << distance_min_index_gpu[point]  << " (" << distance_min_gpu[point] << ")" << std::endl;
//
////  for(int i = 0 ; i < 10*5*BLOCKS ; i++)
////    std::cout << distance_index_array[i] << " " << distance_array[i] << std::endl;
////
//  return 1;
//}
