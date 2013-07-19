#include <strings.h>
#include <thread>
#include <omp.h>
//
// UCSF
//
#include "Build_labeled_domain.h"
#include "enum.h"
#include "Labeled_domain.h"
#include "VTK_implicite_domain.h"
#include "Access_parameters.h"
//
// VTK
//
#include <vtkSmartPointer.h>
#include <vtkTimerLog.h>
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Image_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Mesh_3/Image_to_labeled_function_wrapper.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
// Implicite functions
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh_default_triangulation_3 Triangle_surface;
typedef Triangle_surface::Geom_traits GT;
typedef CGAL::Mesh_3::Image_to_labeled_function_wrapper<CGAL::Image_3, Kernel > Image_wrapper;
//
// We give a comprehensive type name
//
typedef Domains::Build_labeled_domain Domains_Build_labeled;
typedef Domains::Access_parameters DAp;

#ifdef DEBUG_UCSF
// Time log
vtkSmartPointer<vtkTimerLog> timerLog = 
  vtkSmartPointer<vtkTimerLog>::New();
#endif

//
//
//
Domains_Build_labeled::Build_labeled_domain()
{  
  //
  // Header image's information MUST be the same for eigen values et vectores
  number_of_pixels_x_ = (DAp::get_instance())->get_number_of_pixels_x_();
  number_of_pixels_y_ = (DAp::get_instance())->get_number_of_pixels_y_();
  number_of_pixels_z_ = (DAp::get_instance())->get_number_of_pixels_z_();
  //
  size_of_pixel_size_x_ = (DAp::get_instance())->get_number_of_pixels_x_();
  size_of_pixel_size_y_ = (DAp::get_instance())->get_number_of_pixels_y_();
  size_of_pixel_size_z_ = (DAp::get_instance())->get_number_of_pixels_z_();
  //
  rotation_    = (DAp::get_instance())->get_rotation_();
  translation_ = (DAp::get_instance())->get_translation_();
  //
  qfac_ = (int)rotation_.determinant();
  if ( !(qfac_ == 1 || qfac_ == -1) )
    {
      std::cerr << "qfac = " << qfac_ << endl;
      std::cerr << "qfac value must be -1 or 1." << endl;
      abort();
    }

  //
  // Data initialization
  //
  file_inrimage_ = new std::ofstream("head_model.inr", std::ios::out | std::ios::binary);
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
//Domains_Build_labeled::Build_labeled_domain( const Domains_Build_labeled& that )
//{
//}
////
////
////
//Domains_Build_labeled::Build_labeled_domain( Domains_Build_labeled&& that )
//{
//}
//
//
//
Domains_Build_labeled::~Build_labeled_domain()
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
//Domains_Build_labeled& 
//Domains_Build_labeled::operator = ( const Domains_Build_labeled& that )
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
//Domains_Build_labeled& 
//Domains_Build_labeled::operator = ( Domains_Build_labeled&& that )
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
Domains_Build_labeled::operator()()
{
  Head_model_segmentation();
  Write_inrimage_file();
}
//
//
//
void
Domains_Build_labeled::Head_model_segmentation()
{
#ifdef DEBUG_UCSF
  timerLog->GetUniversalTime();
#endif
  
  //
  // Skull and scalp
#ifdef DEBUG_UCSF
  timerLog->MarkEvent("Skull and scalp");
#endif
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    outside_scalp( (DAp::get_instance())->get_outer_skin_surface_() );
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    outside_skull( (DAp::get_instance())->get_outer_skull_surface_() );
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    inside_skull( (DAp::get_instance())->get_inner_skull_surface_() );
  //  
//  outside_scalp( data_position_ );
//  outside_skull( data_position_ );
//  inside_skull ( data_position_ );
  std::thread outside_scalp_thread(std::ref(outside_scalp), data_position_);
  std::thread outside_skull_thread(std::ref(outside_skull), data_position_);
  std::thread inside_skull_thread (std::ref(inside_skull), data_position_);
  outside_scalp_thread.join();
  outside_skull_thread.join();
  inside_skull_thread.join();

  //
  // Cortical segmentation
#ifdef DEBUG_UCSF
  timerLog->MarkEvent("Cortical segmentation");
#endif
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    left_gray_matter ( (DAp::get_instance())->get_lh_pial_() );
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    right_gray_matter ( (DAp::get_instance())->get_rh_pial_() );
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    left_white_matter ( (DAp::get_instance())->get_lh_smoothwm_() );
  Labeled_domain< VTK_implicite_domain, GT::Point_3 > 
    right_white_matter ( (DAp::get_instance())->get_rh_smoothwm_() );
  //
//  left_gray_matter( data_position_ );
//  right_gray_matter( data_position_ );
//  left_white_matter( data_position_ );
//  right_white_matter( data_position_ );
  std::thread left_gray_matter_thread(std::ref(left_gray_matter), data_position_);
  std::thread right_gray_matter_thread(std::ref(right_gray_matter), data_position_);
  std::thread left_white_matter_thread(std::ref(left_white_matter), data_position_);
  std::thread right_white_matter_thread(std::ref(right_white_matter), data_position_);
  left_gray_matter_thread.join();
  right_gray_matter_thread.join();
  left_white_matter_thread.join();
  right_white_matter_thread.join();

  //
  // Subcortical segmenation
  // FreeSurfer output aseg.mgz
#ifdef DEBUG_UCSF
  timerLog->MarkEvent("Subcortical segmenation");
#endif
  CGAL::Image_3 aseg;
  aseg.read( (DAp::get_instance())->get_aseg_hdr_() );
  Image_wrapper subcortical_brain ( aseg );

  //
  // main loop building inrimage data
#ifdef DEBUG_UCSF
  timerLog->MarkEvent("building inrimage data");
#endif
  Eigen::Matrix< float, 3, 1 > position;
  // speed-up
  bool is_in_CSF = false;
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
	  //	  std::cout << position << std::endl << std::endl;
	  //
	  GT::Point_3 cell_center_aseg(data_position_[idx][0],
				       data_position_[idx][1],
				       data_position_[idx][2]);

	  //
	  // Brain segmentation
	  //
	  
	  //
	  // Scalp and skull
	  if( outside_scalp.inside_domain( cell_center ) ) 
	    data_label_[ idx ] = OUTSIDE_SCALP; 
	  //
	  if( outside_skull.inside_domain( cell_center ) ) 
	    data_label_[ idx ] = OUTSIDE_SKULL; 
	  //
	  if( inside_skull.inside_domain( cell_center ) ||
	      subcortical_brain(cell_center_aseg) == CSF )
	    {
	      data_label_[ idx ] = CEREBROSPINAL_FLUID; 
	      is_in_CSF = true;
	    }
	  else
	    is_in_CSF = false;

	  if ( is_in_CSF ) 
	    {
	      //
	      // Cortex
	      if( right_gray_matter.inside_domain( cell_center ) || 
		  left_gray_matter.inside_domain( cell_center ) )
		data_label_[ idx ] = GRAY_MATTER;
	      //
	      //	  if( subcortical_brain(cell_center_aseg) == RIGHT_CEREBRAL_CORTEX || 
	      //	      subcortical_brain(cell_center_aseg) == LEFT_CEREBRAL_CORTEX  )
	      //	    data_label[ idx ] = GRAY_MATTER;

	      //
	      // White matter
	      if( right_white_matter.inside_domain( cell_center ) || 
		  left_white_matter.inside_domain( cell_center ) )
		data_label_[ idx ] = WHITE_MATTER;
	      //
	      //	  if( subcortical_brain(cell_center_aseg) == WM_HYPOINTENSITIES          || 
	      //	      subcortical_brain(cell_center_aseg) == RIGHT_CEREBRAL_WHITE_MATTER || 
	      //	      subcortical_brain(cell_center_aseg) ==  LEFT_CEREBRAL_WHITE_MATTER )
	      //	    data_label[ idx ] = WHITE_MATTER;

	      //
	      // Cerebellum
	      if( subcortical_brain(cell_center_aseg) == RIGHT_CEREBELLUM_CORTEX || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_CEREBELLUM_CORTEX )
		data_label_[ idx ] = CEREBELLUM_GRAY_MATTER;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_CEREBELLUM_WHITE_MATTER || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_CEREBELLUM_WHITE_MATTER )
		data_label_[ idx ] = CEREBELLUM_WHITE_MATTER;

	      //
	      // Subcortex
	      if( subcortical_brain(cell_center_aseg) == BRAIN_STEM )
		data_label_[ idx ] = BRAIN_STEM_SUBCORTICAL;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_HIPPOCAMPUS || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_HIPPOCAMPUS )
		data_label_[ idx ] = HIPPOCAMPUS;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_AMYGDALA || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_AMYGDALA )
		data_label_[ idx ] = AMYGDALA_SUBCORTICAL;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_CAUDATE || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_CAUDATE )
		data_label_[ idx ] = CAUDATE;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_PUTAMEN || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_PUTAMEN )
		data_label_[ idx ] = PUTAMEN;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_THALAMUS_PROPER || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_THALAMUS_PROPER )
		data_label_[ idx ] = THALAMUS;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_ACCUMBENS_AREA || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_ACCUMBENS_AREA )
		data_label_[ idx ] = ACCUMBENS;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_PALLIDUM || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_PALLIDUM )
		data_label_[ idx ] = PALLIDUM;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_LATERAL_VENTRICLE || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_LATERAL_VENTRICLE || 
		  subcortical_brain(cell_center_aseg) == RIGHT_INF_LAT_VENT      || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_INF_LAT_VENT      ||
		  subcortical_brain(cell_center_aseg) == FOURTH_VENTRICLE        || 
		  subcortical_brain(cell_center_aseg) == THIRD_VENTRICLE         || 
		  subcortical_brain(cell_center_aseg) == FIFTH_VENTRICLE         )
		data_label_[ idx ] = CSF;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_CHOROID_PLEXUS || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_CHOROID_PLEXUS )
		data_label_[ idx ] = PLEXUS;
	      //
	      if( subcortical_brain(cell_center) ==  FORNIX )
		data_label_[ idx ] = FORNIX_SUBCORTICAL;
	      //
	      if( subcortical_brain(cell_center_aseg) == CC_POSTERIOR     || 
		  subcortical_brain(cell_center_aseg) == CC_MID_POSTERIOR || 
		  subcortical_brain(cell_center_aseg) == CC_CENTRAL       || 
		  subcortical_brain(cell_center_aseg) == CC_MID_ANTERIOR  || 
		  subcortical_brain(cell_center_aseg) == CC_ANTERIOR      )
		data_label_[ idx ] = CORPUS_COLLOSUM;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_VESSEL || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_VESSEL )
		data_label_[ idx ] = VESSEL;
	      //
	      if( subcortical_brain(cell_center_aseg) == RIGHT_VENTRALDC || 
		  subcortical_brain(cell_center_aseg) ==  LEFT_VENTRALDC )
		data_label_[ idx ] = VENTRAL_DIENCEPHALON;
	      //
	      if( subcortical_brain(cell_center_aseg) == OPTIC_CHIASM )
		data_label_[ idx ] = OPTIC_CHIASM_SUBCORTICAL;
	    }
	}
 
 //
 //
#ifdef DEBUG_UCSF
 //
 // Time log 
 std::cout << "Build_labeled_domain - event log:" << *timerLog << std::endl;
#endif
}
//
//
//
void
Domains_Build_labeled::Build_mesh()
{
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const Domains_Build_labeled& that)
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
