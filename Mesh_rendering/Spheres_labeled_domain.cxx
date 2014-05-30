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
#include <strings.h>
#include <thread>
#include <omp.h>
//
// UCSF
//
#include "Spheres_labeled_domain.h"
#include "Utils/enum.h"
#include "Labeled_domain.h"
#include "Spheres_implicite_domain.h"
#include "Access_parameters.h"
#include "Point_vector.h"
#include "Build_electrodes_list.h"
//
// VTK
//
#include <vtkSmartPointer.h>
#include <vtkTimerLog.h>
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Mesh_3/Image_to_labeled_function_wrapper.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
// Implicite functions
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;
typedef CGAL::Surface_mesh_default_triangulation_3 Triangle_surface;
typedef Triangle_surface::Geom_traits GT;
typedef CGAL::Mesh_3::Image_to_labeled_function_wrapper<CGAL::Image_3, Kernel > Image_wrapper;
//
// We give a comprehensive type name
//
typedef Domains::Spheres_labeled_domain Domains_Spheres_labeled;
typedef Domains::Access_parameters DAp;

#ifdef DEBUG_UCSF
//// Time log
//vtkSmartPointer<vtkTimerLog> timerLog = 
//  vtkSmartPointer<vtkTimerLog>::New();
#endif

//
//
//
Domains_Spheres_labeled::Spheres_labeled_domain()
{  
  //
  // Transformation
  //
  // Reset of transformation matrices
  (Domains::Access_parameters::get_instance())->set_rotation_MNI305_();
  (Domains::Access_parameters::get_instance())->set_translation_MNI305_();
  //
  rotation_    = (DAp::get_instance())->get_rotation_();
  translation_ = (DAp::get_instance())->get_translation_();


  //
  // Data initialization
  //
  std::string head_model_inr = (Domains::Access_parameters::get_instance())->get_files_path_output_();
  head_model_inr += std::string("head_model.inr");
  file_inrimage_ = new std::ofstream(head_model_inr.c_str(), std::ios::out | std::ios::binary);
  data_label_    = new char[ 256 * 256 * 256 ];
  data_position_ = new double*[ 256 * 256 * 256 ];

  //
  // initialisation 256 * 256 * 256 voxels set at 0
  for ( int k = 0; k < 256; k++ )
    for ( int j = 0; j < 256; j++ )
      for ( int i = 0; i < 256; i++ )
	{
	  int idx = i + j*256 + k*256*256;
	  data_label_[idx] = NO_SEGMENTATION;
	  // center of the voxel centered into the MRI framework
	  data_position_[idx] = new double[3];
	  data_position_[idx][0] = i + 0.5; 
	  data_position_[idx][1] = j + 0.5;  
	  data_position_[idx][2] = k + 0.5;
	}

  // 
  //Create the INRIMAGE
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
  file_inrimage_->write(  header.c_str(), header.size() );
}
////
////
////
//Domains_Spheres_labeled::Spheres_labeled_domain( const Domains_Spheres_labeled& that )
//{
//}
////
////
////
//Domains_Spheres_labeled::Spheres_labeled_domain( Domains_Spheres_labeled&& that )
//{
//}
//
//
//
Domains_Spheres_labeled::~Spheres_labeled_domain()
{  
  // close INRIMAGE file
  file_inrimage_->close();
  delete file_inrimage_;

  //
  // clean-up the arrays
  delete [] data_label_;
  //
  for ( int k = 0; k < 256; k++ )
    for ( int j = 0; j < 256; j++ )
      for ( int i = 0; i < 256; i++ )
	delete [] data_position_[i + j*256 + k*256*256];
  delete [] data_position_;

}
////
////
////
//Domains_Spheres_labeled& 
//Domains_Spheres_labeled::operator = ( const Domains_Spheres_labeled& that )
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
//Domains_Spheres_labeled& 
//Domains_Spheres_labeled::operator = ( Domains_Spheres_labeled&& that )
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
Domains_Spheres_labeled::operator()()
{
  model_segmentation();
  write_inrimage_file();
}
//
//
//
void
Domains_Spheres_labeled::model_segmentation()
{
  //
  // brain, CSF, scalp and skull
  Labeled_domain< Spheres_implicite_domain, GT::Point_3, std::list< Point_vector > > 
    scalp( "../share/sphere_scalp.xyz" );
  Labeled_domain< Spheres_implicite_domain, GT::Point_3, std::list< Point_vector > > 
    skull( "../share/sphere_skull.xyz" );
  Labeled_domain< Spheres_implicite_domain, GT::Point_3, std::list< Point_vector > > 
    CSF( "../share/sphere_CSF.xyz"  );
  Labeled_domain< Spheres_implicite_domain, GT::Point_3, std::list< Point_vector > > 
    brain( "../share/sphere_brain.xyz" );
  //  
//    scalp( data_position_ );
//    skull( data_position_ );
//    CSF  ( data_position_ );
//    brain( data_position_ );
  std::thread scalp_thread(std::ref(scalp), data_position_);
  std::thread skull_thread(std::ref(skull), data_position_);
  std::thread CSF_thread  (std::ref(CSF),   data_position_);
  std::thread brain_thread(std::ref(brain), data_position_);
  // End of spheres segmentation
  scalp_thread.join();
  skull_thread.join();
  CSF_thread.join();
  brain_thread.join();
  
  //
  // Electrodes localization
  Domains::Build_electrodes_list electrodes;
  electrodes.adjust_cap_positions_on( scalp );
  // Write electrodes XML file
  electrodes.Output_electrodes_list_xml();

  //
  // main loop building inrimage data
  // speed-up
  bool is_in_CSF = false;
  Eigen::Matrix< float, 3, 1 > position;
  //
  // create a data_label_tmp private in the different
  for ( int k = 0; k < 256; k++ )
    for ( int j = 0; j < 256; j++ )
      for ( int i = 0; i < 256; i++ )
	{
 	  int idx = i + j*256 + k*256*256;
	  //
	  position <<
	    /* size_of_pixel_size_x_ * */ data_position_[idx][0],
	    /* size_of_pixel_size_y_ * */ data_position_[idx][1],
	    /* size_of_pixel_size_z_ * */ data_position_[idx][2];
	  position = rotation_ * position + translation_;
	  //
	  GT::Point_3 cell_center(position(0,0),
				  position(1,0),
				  position(2,0));

	  //
	  // Electrodes position
	  //
	  if( electrodes.inside_domain( cell_center ) ) 
	    data_label_[ idx ] = ELECTRODE; 
	  

	  //
	  // Brain segmentation
	  //
	  
	  //
	  // Scalp and skull
	  if( scalp.inside_domain( cell_center ) ) 
	    data_label_[ idx ] = OUTSIDE_SCALP; 
	  //
	  if( skull.inside_domain( cell_center ) ) 
	    data_label_[ idx ] = OUTSIDE_SKULL; 
	  //
	  if( CSF.inside_domain( cell_center ) )
	    {
	      data_label_[ idx ] = CEREBROSPINAL_FLUID; 
	      is_in_CSF = true;
	    }
	  else
	    is_in_CSF = false;

	  if ( is_in_CSF ) 
	    {
	      //
	      // Gray-matter
	      if( brain.inside_domain( cell_center ) )
		data_label_[ idx ] = GRAY_MATTER;
	    }
	}
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const Domains_Spheres_labeled& that)
{
  // std::for_each( that.get_list_position().begin(),
  //		 that.get_list_position().end(),
  //		 [&stream]( int Val )
  //		 {
  //		   stream << "list pos = " << Val << "\n";
  //		 });
  // //
  // stream << "position x = " <<    that.get_pos_x() << "\n";
  // stream << "position y = " <<    that.get_pos_y() << "\n";
  // if ( &that.get_tab() )
  //   {
  //     stream << "tab[0] = "     << ( &that.get_tab() )[0] << "\n";
  //     stream << "tab[1] = "     << ( &that.get_tab() )[1] << "\n";
  //     stream << "tab[2] = "     << ( &that.get_tab() )[2] << "\n";
  //     stream << "tab[3] = "     << ( &that.get_tab() )[3] << "\n";
  //   }
  //
  return stream;
};
