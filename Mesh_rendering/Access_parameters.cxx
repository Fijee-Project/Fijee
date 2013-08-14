#include "Access_parameters.h"
//
// We give a comprehensive type name
//
typedef Domains::Access_parameters DAp;
//
//
//
DAp*
DAp::parameters_instance_ = NULL;
//
//
//
DAp::Access_parameters():
  conductivity_tensors_array_(nullptr),
  eigen_values_matrices_array_(nullptr),
  positions_array_(nullptr),
  Do_we_have_conductivity_(nullptr)
{
  //
  // Check on ENV variables
  char
    *subjects_dir = getenv ("SUBJECTS_DIR"),
    *subject      = getenv ("SUBJECT");
  //
  if( subjects_dir == NULL ||
      subject      == NULL )
    {
      std::cerr << "FreeSurfer Env variables $SUBJECTS_DIR and $SUBJECT must be defined" 
		<< std::endl;
      exit(1);
    }
  //
  files_path_ = std::string(subjects_dir) + "/" + std::string(subject) + "/";

  //
  // Mesh rendering
  //

  //
  // Skull and scalp
  std::string skull_scalp_files_path = files_path_ + "/bem/watershed/STL/" + std::string(subject);
  outer_skin_surface_  = skull_scalp_files_path + "_outer_skin_surface.stl";
  outer_skull_surface_ = skull_scalp_files_path + "_outer_skull_surface.stl";
  inner_skull_surface_ = skull_scalp_files_path + "_inner_skull_surface.stl";

  //
  // Cortical segmentation
  std::string cortical_files_path = files_path_ + "/surf/STL/";
  lh_pial_     = cortical_files_path + "lh.pial.stl";
  rh_pial_     = cortical_files_path + "rh.pial.stl";
  lh_smoothwm_ = cortical_files_path + "lh.smoothwm.stl";
  rh_smoothwm_ = cortical_files_path + "rh.smoothwm.stl";

  //
  // Subcortical segmenation
  aseg_hdr_ = files_path_ + "mri/m000-aseg.hdr";

  //
  // NIFTI information on all segmentation file
  //
  
  //
  // All segmentation nifti file (aseg.nii)
  std::string check_header = files_path_ + "mri/aseg.nii";
  FILE *aseg_file_nifti;
  aseg_file_nifti = fopen( check_header.c_str(),  "r");
  if ( aseg_file_nifti == NULL ) {
    fprintf(stderr, "\n Error opening header file %s \n", check_header.c_str());
    exit(1);
  }

  //
  // nifti headers and files
  nifti_1_header aseg_header_nifti;
  // MIN_HEADER_SIZE 348
  int ret_nifti = fread( &aseg_header_nifti , MIN_HEADER_SIZE, 1, aseg_file_nifti );
  if ( ret_nifti != 1 ) {
    fprintf(stderr, "\nError reading header file %s\n", check_header.c_str());
    exit(1);
  }

  //
  // Header image's information MUST be the same for eigen values et vectores
  number_of_pixels_x_ = aseg_header_nifti.dim[1];
  number_of_pixels_y_ = aseg_header_nifti.dim[2];
  number_of_pixels_z_ = aseg_header_nifti.dim[3];
  //
  size_of_pixel_size_x_ = aseg_header_nifti.pixdim[1];
  size_of_pixel_size_y_ = aseg_header_nifti.pixdim[2];
  size_of_pixel_size_z_ = aseg_header_nifti.pixdim[3];
  //
  rotation_ << 
    aseg_header_nifti.srow_x[0], aseg_header_nifti.srow_x[1], aseg_header_nifti.srow_x[2],
    aseg_header_nifti.srow_y[0], aseg_header_nifti.srow_y[1], aseg_header_nifti.srow_y[2],
    aseg_header_nifti.srow_z[0], aseg_header_nifti.srow_z[1], aseg_header_nifti.srow_z[2];
  //
  translation_ <<
    aseg_header_nifti.srow_x[3],
    aseg_header_nifti.srow_y[3],
    aseg_header_nifti.srow_z[3];

  //
  // Some print out
  //

#ifdef TRACE
#if ( TRACE == 1 )
  /********** print a little header information */
  fprintf(stderr, "\n %s header information for:", check_header.c_str() );
  fprintf(stderr, "\n XYZT dimensions: %d %d %d %d", 
	  number_of_pixels_x_,number_of_pixels_y_,number_of_pixels_z_,aseg_header_nifti.dim[4]);
  fprintf(stderr, "\n Datatype code and bits/pixel: %d %d",aseg_header_nifti.datatype,aseg_header_nifti.bitpix);
  fprintf(stderr, "\n Scaling slope and intercept: %.6f %.6f",aseg_header_nifti.scl_slope,aseg_header_nifti.scl_inter);
  fprintf(stderr, "\n Byte offset to data in datafile: %ld",(long)(aseg_header_nifti.vox_offset));
  fprintf(stderr, "\n qform_code: %.d",(aseg_header_nifti.qform_code));
  fprintf(stderr, "\n sform_code: %.d",(aseg_header_nifti.sform_code));
  fprintf(stderr, "\n");
  std::cout << "Rotation matrix" << std::endl;
  std::cout << rotation_ << std::endl;
  std::cout << "Rotation matrix determinant" << std::endl;
  std::cout << rotation_.determinant() << std::endl;
  std::cout << "Translation vector" << std::endl;
  std::cout << translation_ << std::endl;
#endif
#endif

  //
  // NIFTI information on diffusion data
  //

  //
  // All segmentation nifti file (aseg.nii)
  std::string nifti_eigen_values  = files_path_ + "/Diffusion_tensor/eigvals.nii";
  std::string nifti_eigen_vector1 = files_path_ + "/Diffusion_tensor/eigvec1.nii";
  std::string nifti_eigen_vector2 = files_path_ + "/Diffusion_tensor/eigvec2.nii";
  std::string nifti_eigen_vector3 = files_path_ + "/Diffusion_tensor/eigvec3.nii";

  //
  // nifti headers and files
  nifti_1_header header_eigenvalues;
  nifti_1_header header_eigenvector1;
  nifti_1_header header_eigenvector2;
  nifti_1_header header_eigenvector3;
  //
  FILE *file_eigen_values;
  FILE *file_eigen_vector1;
  FILE *file_eigen_vector2;
  FILE *file_eigen_vector3;

  //
  // Open and read nifti header and retrieve header information
  file_eigen_values  = fopen( nifti_eigen_values.c_str(),  "r");
  file_eigen_vector1 = fopen( nifti_eigen_vector1.c_str(), "r");
  file_eigen_vector2 = fopen( nifti_eigen_vector2.c_str(), "r");
  file_eigen_vector3 = fopen( nifti_eigen_vector3.c_str(), "r");
  if (file_eigen_values == NULL) {
    fprintf(stderr, "\n Error opening header file %s \n", nifti_eigen_values.c_str());
    exit(1);
  }
  if (file_eigen_vector1 == NULL) {
    fprintf(stderr, "\n Error opening header file %s \n", nifti_eigen_vector1.c_str());
    exit(1);
  }
  if (file_eigen_vector2 == NULL) {
    fprintf(stderr, "\n Error opening header file %s \n", nifti_eigen_vector1.c_str());
    exit(1);
  }
  if (file_eigen_vector3 == NULL) {
    fprintf(stderr, "\n Error opening header file %s \n", nifti_eigen_vector1.c_str());
    exit(1);
  }

  //
  // Read the nifti file
  int ret_eigenvalues = fread(&header_eigenvalues, MIN_HEADER_SIZE, 1, file_eigen_values);
  if (ret_eigenvalues != 1) {
    fprintf(stderr, "\nError reading header file %s\n", nifti_eigen_values.c_str());
    exit(1);
  }
  int ret_eigenvector1 = fread(&header_eigenvector1, MIN_HEADER_SIZE, 1, file_eigen_vector1);
  if (ret_eigenvector1 != 1) {
    fprintf(stderr, "\nError reading header file %s\n", nifti_eigen_vector1.c_str());
    exit(1);
  }
  int ret_eigenvector2 = fread(&header_eigenvector2, MIN_HEADER_SIZE, 1, file_eigen_vector2);
  if (ret_eigenvector2 != 1) {
    fprintf(stderr, "\nError reading header file %s\n", nifti_eigen_vector2.c_str());
    exit(1);
  }
  int ret_eigenvector3 = fread(&header_eigenvector3, MIN_HEADER_SIZE, 1, file_eigen_vector3);
  if (ret_eigenvector3 != 1) {
    fprintf(stderr, "\nError reading header file %s\n", nifti_eigen_vector3.c_str());
    exit(1);
  }

  //
  // Header image's information MUST be the same for eigen values et vectores
  eigenvalues_number_of_pixels_x_ = header_eigenvalues.dim[1];
  eigenvalues_number_of_pixels_y_ = header_eigenvalues.dim[2];
  eigenvalues_number_of_pixels_z_ = header_eigenvalues.dim[3];
  eigenvalues_number_of_layers_   = header_eigenvalues.dim[4];
  //
  eigenvalues_size_of_pixel_size_x_ = header_eigenvalues.pixdim[1];
  eigenvalues_size_of_pixel_size_y_ = header_eigenvalues.pixdim[2];
  eigenvalues_size_of_pixel_size_z_ = header_eigenvalues.pixdim[3];

  //
  // Rotation Matrix and translation vector
  // The rotation matrix is based on the METHOD 3 of the nifti file (nifti1.h)
  eigenvalues_rotation_ << 
    header_eigenvalues.srow_x[0], header_eigenvalues.srow_x[1], header_eigenvalues.srow_x[2], 
    header_eigenvalues.srow_y[0], header_eigenvalues.srow_y[1], header_eigenvalues.srow_y[2], 
    header_eigenvalues.srow_z[0], header_eigenvalues.srow_z[1], header_eigenvalues.srow_z[2]; 
  //
  eigenvalues_translation_ <<  
    header_eigenvalues.srow_x[3], 
    header_eigenvalues.srow_y[3], 
    header_eigenvalues.srow_z[3];

#ifdef TRACE
#if ( TRACE == 1 )
  /********** print a little header information */
  fprintf(stderr, "\n%s header information for eigen values:", nifti_eigen_values.c_str() );
  fprintf(stderr, "\nXYZT dimensions: %d %d %d %d",
	  eigenvalues_number_of_pixels_x_,
	  eigenvalues_number_of_pixels_y_,
	  eigenvalues_number_of_pixels_z_,
	  eigenvalues_number_of_layers_);
  fprintf(stderr, "\n%s header information for eigen values:", nifti_eigen_vector1.c_str() );
  fprintf(stderr, "\nXYZT dimensions: %d %d %d %d",
	  header_eigenvector1.dim[1],
	  header_eigenvector1.dim[2],
	  header_eigenvector1.dim[3],
	  header_eigenvector1.dim[4]);
   fprintf(stderr, "\nDatatype code and bits/pixel: %d %d",header_eigenvalues.datatype,header_eigenvalues.bitpix);
  fprintf(stderr, "\nScaling slope and intercept: %.6f %.6f",header_eigenvalues.scl_slope,header_eigenvalues.scl_inter);
  fprintf(stderr, "\nByte offset to data in datafile: %ld",(long)(header_eigenvalues.vox_offset));
  fprintf(stderr, "\n");
  //
  std::cout << "Rotation matrix method 3" 
	    << std::endl;
  std::cout << eigenvalues_rotation_ 
	    << std::endl;
  std::cout << "Not an orthogonal matrix: rotation matrix determinant method 3" 
	    << std::endl;
  std::cout << eigenvalues_rotation_.determinant() 
	    << std::endl;
  std::cout << "Translation vector" 
	    << std::endl;
  std::cout << eigenvalues_translation_ 
	    << std::endl;
  //Coordinates
  // x = srow_x[0] * i + srow_x[1] * j + srow_x[2] * k + srow_x[3]
  // y = srow_y[0] * i + srow_y[1] * j + srow_y[2] * k + srow_y[3]
  // z = srow_z[0] * i + srow_z[1] * j + srow_z[2] * k + srow_z[3]
#endif
#endif

  //
  // Read the nifti files
  //

  //
  // open the datafile, jump to data offset
  ret_eigenvalues = fseek(file_eigen_values, (long)(header_eigenvalues.vox_offset), SEEK_SET);
  if (ret_eigenvalues != 0) {
    fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",
	    (long)(header_eigenvalues.vox_offset), nifti_eigen_values.c_str());
    exit(1);
  }
  ret_eigenvector1 = fseek(file_eigen_vector1, (long)(header_eigenvector1.vox_offset), SEEK_SET);
  if (ret_eigenvector1 != 0) {
    fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",
	    (long)(header_eigenvector1.vox_offset), nifti_eigen_vector1.c_str());
    exit(1);
  }
  ret_eigenvector2 = fseek(file_eigen_vector2, (long)(header_eigenvector2.vox_offset), SEEK_SET);
  if (ret_eigenvector2 != 0) {
    fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",
	    (long)(header_eigenvector2.vox_offset), nifti_eigen_vector2.c_str());
    exit(1);
  }
  ret_eigenvector3 = fseek(file_eigen_vector3, (long)(header_eigenvector3.vox_offset), SEEK_SET);
  if (ret_eigenvector3 != 0) {
    fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",
	    (long)(header_eigenvector3.vox_offset), nifti_eigen_vector3.c_str());
    exit(1);
  }

  //
  // Allocate buffer and read 4D volume from data file
  data_eigen_values_ = new float[ eigenvalues_number_of_pixels_x_ * 
				  eigenvalues_number_of_pixels_y_ * 
				  eigenvalues_number_of_pixels_z_ * 
				  eigenvalues_number_of_layers_ ];
  if ( data_eigen_values_ == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer for %s\n", nifti_eigen_values.c_str());
    exit(1);
  }
  data_eigen_vector1_ = new float[ eigenvalues_number_of_pixels_x_ * 
				   eigenvalues_number_of_pixels_y_ * 
				   eigenvalues_number_of_pixels_z_ * 
				   eigenvalues_number_of_layers_ ];
  if ( data_eigen_vector1_ == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer for %s\n", nifti_eigen_vector1.c_str());
    exit(1);
  }
  data_eigen_vector2_ = new float[ eigenvalues_number_of_pixels_x_ * 
				   eigenvalues_number_of_pixels_y_ * 
				   eigenvalues_number_of_pixels_z_ * 
				   eigenvalues_number_of_layers_ ];
  if ( data_eigen_vector2_ == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer for %s\n", nifti_eigen_vector2.c_str());
    exit(1);
  }
  data_eigen_vector3_ = new float[ eigenvalues_number_of_pixels_x_ * 
				   eigenvalues_number_of_pixels_y_ * 
				   eigenvalues_number_of_pixels_z_ * 
				   eigenvalues_number_of_layers_ ];
  if ( data_eigen_vector3_ == nullptr ) {
    fprintf(stderr, "\nError allocating data buffer for %s\n", nifti_eigen_vector3.c_str());
    exit(1);
  }
  //
  ret_eigenvalues = fread( data_eigen_values_, sizeof(float), 
			   eigenvalues_number_of_pixels_x_ * 
			   eigenvalues_number_of_pixels_y_ * 
			   eigenvalues_number_of_pixels_z_ * 
			   eigenvalues_number_of_layers_, file_eigen_values);
  if ( ret_eigenvalues != eigenvalues_number_of_pixels_x_ * eigenvalues_number_of_pixels_y_ * eigenvalues_number_of_pixels_z_ * eigenvalues_number_of_layers_ ) {
    fprintf(stderr, "\nError reading volume 1 from %s (%d) \n", nifti_eigen_values.c_str(), ret_eigenvalues);
    exit(1);
  }
  ret_eigenvector1 = fread( data_eigen_vector1_, sizeof(float), 
			    eigenvalues_number_of_pixels_x_ * 
			    eigenvalues_number_of_pixels_y_ * 
			    eigenvalues_number_of_pixels_z_ * 
			    eigenvalues_number_of_layers_, file_eigen_vector1);
  if ( ret_eigenvector1 != eigenvalues_number_of_pixels_x_ * eigenvalues_number_of_pixels_y_ * eigenvalues_number_of_pixels_z_ * eigenvalues_number_of_layers_ ) {
    fprintf(stderr, "\nError reading volume 1 from %s (%d)\n", nifti_eigen_vector1.c_str(), ret_eigenvector1);
    exit(1);
  }
  ret_eigenvector2 = fread( data_eigen_vector2_, sizeof(float), 
			    eigenvalues_number_of_pixels_x_ * 
			    eigenvalues_number_of_pixels_y_ * 
			    eigenvalues_number_of_pixels_z_ * 
			    eigenvalues_number_of_layers_, file_eigen_vector2);
  if ( ret_eigenvector2 != eigenvalues_number_of_pixels_x_ * eigenvalues_number_of_pixels_y_ * eigenvalues_number_of_pixels_z_ * eigenvalues_number_of_layers_ ) {
    fprintf(stderr, "\nError reading volume 1 from %s (%d)\n", nifti_eigen_vector2.c_str(), ret_eigenvector2);
    exit(1);
  }
  ret_eigenvector3 = fread( data_eigen_vector3_, sizeof(float), 
			    eigenvalues_number_of_pixels_x_ * 
			    eigenvalues_number_of_pixels_y_ * 
			    eigenvalues_number_of_pixels_z_ * 
			    eigenvalues_number_of_layers_, file_eigen_vector3);
  if ( ret_eigenvector3 != eigenvalues_number_of_pixels_x_ * eigenvalues_number_of_pixels_y_ * eigenvalues_number_of_pixels_z_ * eigenvalues_number_of_layers_ ) {
    fprintf(stderr, "\nError reading volume 1 from %s (%d)\n", nifti_eigen_vector3.c_str(), ret_eigenvector3);
    exit(1);
  }  

  //
  // Clean up
  fclose(aseg_file_nifti);
  fclose(file_eigen_values);
  fclose(file_eigen_vector1);
  fclose(file_eigen_vector2);
  fclose(file_eigen_vector3);
}
//
//
//
DAp::Access_parameters( const DAp& that ){}
//
//
//
DAp::~Access_parameters(){
  //
  //
  if ( data_eigen_values_ != nullptr )
    {
      delete [] data_eigen_values_;
      data_eigen_values_ = nullptr;
    }
  //
  if ( data_eigen_vector1_ != nullptr )
    {
      delete [] data_eigen_vector1_;
      data_eigen_vector1_ = nullptr;
    }
  //
  if ( data_eigen_vector2_ != nullptr )
    {
      delete [] data_eigen_vector2_;
      data_eigen_vector2_ = nullptr;
    }
  //
  if ( data_eigen_vector3_ != nullptr )
    {
      delete [] data_eigen_vector3_;
      data_eigen_vector3_ = nullptr;
    }
  //
  if ( conductivity_tensors_array_ != nullptr )
    {
      delete [] conductivity_tensors_array_;
      conductivity_tensors_array_ = nullptr;
    }
  //
  if ( eigen_values_matrices_array_ != nullptr )
    {
      delete [] eigen_values_matrices_array_;
      eigen_values_matrices_array_ = nullptr;
    }
  //
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
//
//
//
DAp& 
DAp::operator = ( const DAp& that )
{
  //
  return *this;
}
//
//
//
void
DAp::get_data_eigen_values_( float** Data_Eigen_Values )
{
  if( data_eigen_values_ != nullptr )
    {
      if( data_eigen_values_ != *Data_Eigen_Values )
	{
	  *Data_Eigen_Values = data_eigen_values_;
	  data_eigen_values_ = nullptr;
	}
    }
  else
    {
      std::cerr << "data_eigen_values_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_data_eigen_vector1_( float** Data_Eigen_Vector1 )
{
  if( data_eigen_vector1_ != nullptr )
    {
      if( data_eigen_vector1_ != *Data_Eigen_Vector1 )
	{
	  *Data_Eigen_Vector1 = data_eigen_vector1_;
	  data_eigen_vector1_ = nullptr;
	}
    }
  else
    {
      std::cerr << "data_eigen_vector1_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_data_eigen_vector2_( float** Data_Eigen_Vector2 )
{
  if( data_eigen_vector2_ != nullptr )
    {
      if( data_eigen_vector2_ != *Data_Eigen_Vector2 )
	{
	  *Data_Eigen_Vector2 = data_eigen_vector2_;
	  data_eigen_vector2_ = nullptr;
	}
    }
  else
    {
      std::cerr << "data_eigen_vector2_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_data_eigen_vector3_( float** Data_Eigen_Vector3 )
{
  if( data_eigen_vector3_ != nullptr )
    {
      if( data_eigen_vector3_ != *Data_Eigen_Vector3 )
	{
	  *Data_Eigen_Vector3 = data_eigen_vector3_;
	  data_eigen_vector3_ = nullptr;
	}
    }
  else
    {
      std::cerr << "data_eigen_vector3_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_conductivity_tensors_array_(  Eigen::Matrix <float, 3, 3>** Conductivity_Tensors_Array )
{
  if( conductivity_tensors_array_ != nullptr )
    {
      if( conductivity_tensors_array_ != *Conductivity_Tensors_Array )
	{
	  *Conductivity_Tensors_Array = conductivity_tensors_array_;
	  conductivity_tensors_array_ = nullptr;
	}
    }
  else
    {
      std::cerr << "conductivity_tensors_array_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_eigen_values_matrices_array_(  Eigen::Matrix <float, 3, 3>** Eigen_Values_Matrices_Array )
{
  if( eigen_values_matrices_array_ != nullptr )
    {
      if( eigen_values_matrices_array_ != *Eigen_Values_Matrices_Array )
	{
	  *Eigen_Values_Matrices_Array = eigen_values_matrices_array_;
	  eigen_values_matrices_array_ = nullptr;
	}
    }
  else
    {
      std::cerr << "eigen_values_matrices_array_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_positions_array_(  Eigen::Matrix <float, 3, 1>** Positions_Array )
{
  if( positions_array_ != nullptr )
    {
      if( positions_array_ != *Positions_Array )
	{
	  *Positions_Array = positions_array_;
	  positions_array_ = nullptr;
	}
    }
  else
    {
      std::cerr << "positions_array_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
void
DAp::get_Do_we_have_conductivity_(  bool** Do_We_Have_Conductivity )
{
  if( Do_we_have_conductivity_ != nullptr )
    {
      if( Do_we_have_conductivity_ != *Do_We_Have_Conductivity )
	{
	  *Do_We_Have_Conductivity = Do_we_have_conductivity_;
	  Do_we_have_conductivity_ = nullptr;
	}
    }
  else
    {
      std::cerr << "Do_we_have_conductivity_ is already transfered" << std::endl;
      abort();
    }
}
//
//
//
DAp* 
DAp::get_instance()
{
  if( parameters_instance_ == NULL )
    parameters_instance_ = new DAp();
  //
  return parameters_instance_;
}
//
//
//
void
DAp::set_conductivity_tensors_array_(  Eigen::Matrix <float, 3, 3>** Conductivity_Tensors_Array )
{
  if ( conductivity_tensors_array_ != nullptr )
    {
      delete [] conductivity_tensors_array_;
      conductivity_tensors_array_ = nullptr;
    }
  //
  conductivity_tensors_array_ = *Conductivity_Tensors_Array;
  *Conductivity_Tensors_Array = nullptr;
}
//
//
//
void
DAp::set_eigen_values_matrices_array_(  Eigen::Matrix <float, 3, 3>** Eigen_Values_Matrices_Array )
{
  if ( eigen_values_matrices_array_ != nullptr )
    {
      delete [] eigen_values_matrices_array_;
      eigen_values_matrices_array_ = nullptr;
    }
  //
  eigen_values_matrices_array_ = *Eigen_Values_Matrices_Array;
  *Eigen_Values_Matrices_Array = nullptr;
}
//
//
//
void
DAp::set_positions_array_(  Eigen::Matrix <float, 3, 1>** Positions_Array )
{
  if ( positions_array_ != nullptr )
    {
      delete [] positions_array_;
      positions_array_ = nullptr;
    }
  //
  positions_array_ = *Positions_Array;
  *Positions_Array = nullptr;
}
//
//
//
void
DAp::set_Do_we_have_conductivity_(  bool** Do_We_Have_Conductivity )
{
  if ( Do_we_have_conductivity_ != nullptr )
    {
      delete [] Do_we_have_conductivity_;
      Do_we_have_conductivity_ = nullptr;
    }
  //
  Do_we_have_conductivity_ = *Do_We_Have_Conductivity;
  *Do_We_Have_Conductivity = nullptr;
}
//
//
//
void 
DAp::kill_instance()
{
  if( parameters_instance_ != NULL )
    {
      delete parameters_instance_;
      parameters_instance_ = NULL;
    }
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
					  const DAp& that)
{
  stream << " Pattern Singleton\n";
  //
  return stream;
}
