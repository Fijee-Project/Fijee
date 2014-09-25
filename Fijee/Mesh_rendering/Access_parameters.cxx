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
#include "Access_parameters.h"
//
// We give a comprehensive type name
//
typedef Domains::Access_parameters DAp;
//typedef struct stat stat_file;
//
//
//
DAp*
DAp::parameters_instance_ = NULL;
//
//
//
DAp::Access_parameters(): Fijee::XML_writer("")
{
  try{
    //
    // Check on ENV variables
    // 
    Fijee::Fijee_environment fijee;
    //
    files_path_        = fijee.get_fem_path_();
    files_path_output_ = fijee.get_fem_output_path_();

    // 
    // Time profiler lof file
    // It the file existes: empty it.
#ifdef TIME_PROFILER
    std::ofstream ofs ( "fijee_time.log", std::ofstream::app );
    if( ofs.good() ) ofs.clear();
    ofs.close();
#endif
  
    // 
    // Read data set
    // 
    std::cout << "Load Fijee data set file" << std::endl;

    // 
    //
    pugi::xml_document     xml_file;
    pugi::xml_parse_result result = xml_file.load_file( "fijee.xml" );
    //
    switch( result.status )
      {
      case pugi::status_ok:
	{
	  //
	  // Check that we have a FIJEE XML file
	  const pugi::xml_node fijee_node = xml_file.child("fijee");
	  if (!fijee_node)
	    {
	      std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
	      exit(1);
	    }
	  
	  //
	  // Get setup node
	  const pugi::xml_node setup_node = fijee_node.child("setup");
	  if (!setup_node)
	    {
	      std::cerr << "Read data from XML: no setup node" << std::endl;
	      exit(1);
	    }
	  // Get install directory
	  install_directory_ = std::string(setup_node.attribute("install_dir").as_string());
	  // 
	  if(install_directory_.empty())
	    {
	      std::string message = std::string("Error reading fijee.xml file.")
		+ std::string(" install_dir flag should be filled up.");
	      //
	      throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	    }
	  // 
	  electrodes_10_20_ = install_directory_ 
	    + std::string("/share/electrodes/electrodes-Standard-10-20-Cap81.xml");

	  
	  //
	  // Get setup node
	  const pugi::xml_node mesh_node = fijee_node.child("mesh");
	  if (!mesh_node)
	    {
	      std::cerr << "Read data from XML: no mesh node." << std::endl;
	      exit(1);
	    }
	  // Graph parcellation: number of parcels
	  number_of_parcels_ = mesh_node.attribute("number_of_parcels").as_int();
	  // 
	  if( number_of_parcels_ == 0 )
	    {
	      std::string message = std::string("Error reading fijee.xml file.")
		+ std::string(" number_of_parcels flag should be filled up.");
	      //
	      throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	    }

	  //
	  //
	  break;
	};
      default:
	{
	  std::string message = std::string("Error reading fijee.xml file.")
	    + std::string(" You should look for an example in the 'share' directory located in the Fijee's install directory.");
	  //
	  throw Fijee::Error_handler( message,  __LINE__, __FILE__ );
	}
      }


    //
    // Mesh rendering
    //

    //
    // Skull and scalp
    std::string skull_scalp_files_path = files_path_ + "fem/input/STL/" + fijee.get_subject_();
    outer_skin_surface_  = skull_scalp_files_path + "_outer_skin_surface.stl";
    outer_skull_surface_ = skull_scalp_files_path + "_outer_skull_surface.stl";
    inner_skull_surface_ = skull_scalp_files_path + "_inner_skull_surface.stl";
    inner_brain_surface_ = skull_scalp_files_path + "_brain_surface.stl";

    //
    // Cortical segmentation
    std::string cortical_files_path = files_path_ + "fem/input/STL/";
    lh_pial_     = cortical_files_path + "lh.pial.stl";
    rh_pial_     = cortical_files_path + "rh.pial.stl";
    lh_smoothwm_ = cortical_files_path + "lh.smoothwm.stl";
    rh_smoothwm_ = cortical_files_path + "rh.smoothwm.stl";

    //
    // Subcortical segmenation
    aseg_hdr_ = files_path_ + "fem/input/mri/aseg.hdr";

    // 
    // SMP input
    sc3T1_ = files_path_ + "fem/input/spm/sc3T1.hdr";
    //
    sc4T1_ = files_path_ + "fem/input/spm/sc4T1.hdr";
    //
    sc5T1_ = files_path_ + "fem/input/spm/sc5T1.hdr";
    //
    sc6T1_ = files_path_ + "fem/input/spm/sc6T1.hdr";


    //
    // NIFTI information on all segmentation file
    //
  
    //
    // All segmentation nifti file (aseg.nii)
    std::string check_header = files_path_ + "fem/input/mri/aseg.nii";
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
    delta_rotation_ << 
      0.,0.,0.,
      0.,0.,0.,
      0.,0.,0.;
    //
    delta_translation_ << 0.,0.,0.;

    //
    // Some print out
    //

#ifdef FIJEE_TRACE
#if ( FIJEE_TRACE == 1 )
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

#ifdef FIJEE_TRACE
#if ( FIJEE_TRACE == 1 )
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
  catch( Fijee::Exception_handler& err )
    {
      std::cerr << err.what() << std::endl;
    }
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
DAp::set_lh_gray_matter_surface_point_normal_(std::list<Point_vector>&&  Lh_gray_matter_surface_point_normal )
{
  lh_gray_matter_surface_point_normal_ = Lh_gray_matter_surface_point_normal;
}
//
//
//
void
DAp::get_lh_gray_matter_surface_point_normal_(std::list<Point_vector>&  Lh_gray_matter_surface_point_normal )const
{
  Lh_gray_matter_surface_point_normal.resize(lh_gray_matter_surface_point_normal_.size());
  // 
  std::copy( lh_gray_matter_surface_point_normal_.begin(),
	     lh_gray_matter_surface_point_normal_.end(),
	     Lh_gray_matter_surface_point_normal.begin() );
}
//
//
//
void
DAp::set_rh_gray_matter_surface_point_normal_(std::list<Point_vector>&&  Rh_gray_matter_surface_point_normal )
{
  rh_gray_matter_surface_point_normal_ = Rh_gray_matter_surface_point_normal;
}
//
//
//
void
DAp::get_rh_gray_matter_surface_point_normal_(std::list<Point_vector>&  Rh_gray_matter_surface_point_normal ) const
{
  Rh_gray_matter_surface_point_normal.resize(rh_gray_matter_surface_point_normal_.size());
  // 
  std::copy( rh_gray_matter_surface_point_normal_.begin(),
	     rh_gray_matter_surface_point_normal_.end(),
	     Rh_gray_matter_surface_point_normal.begin() );
}
//
//
//
void
DAp::set_lh_white_matter_surface_point_normal_(std::list<Point_vector>&&  Lh_white_matter_surface_point_normal )
{
  lh_white_matter_surface_point_normal_ = Lh_white_matter_surface_point_normal;
}
//
//
//
void
DAp::get_lh_white_matter_surface_point_normal_(std::list<Point_vector>&  Lh_white_matter_surface_point_normal )const
{
  Lh_white_matter_surface_point_normal.resize( lh_white_matter_surface_point_normal_.size() );
  // 
  std::copy( lh_white_matter_surface_point_normal_.begin(),
	     lh_white_matter_surface_point_normal_.end(),
	     Lh_white_matter_surface_point_normal.begin() );
}
//
//
//
void
DAp::set_rh_white_matter_surface_point_normal_(std::list<Point_vector>&&  Rh_white_matter_surface_point_normal )
{
  rh_white_matter_surface_point_normal_ = Rh_white_matter_surface_point_normal;
}
//
//
//
void
DAp::get_rh_white_matter_surface_point_normal_(std::list<Point_vector>&  Rh_white_matter_surface_point_normal )const
{
  Rh_white_matter_surface_point_normal.resize(rh_white_matter_surface_point_normal_.size());
  //
  std::copy( rh_white_matter_surface_point_normal_.begin(),
	     rh_white_matter_surface_point_normal_.end(),
	     Rh_white_matter_surface_point_normal.begin() );
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
void 
DAp::epitaxy_growth()
{
  // 
  // Left hemisphere
  match_wm_gm( lh_white_matter_surface_point_normal_,
	       lh_gray_matter_surface_point_normal_,
	       lh_match_wm_gm_);
  // 
  // Right hemisphere
  match_wm_gm( rh_white_matter_surface_point_normal_,
	       rh_gray_matter_surface_point_normal_,
	       rh_match_wm_gm_);
}
//
//
//
void 
DAp::match_wm_gm( std::list<Domains::Point_vector>&  White_matter_surface_point_normal, 
		  std::list<Domains::Point_vector>&  Gray_matter_surface_point_normal, 
		  std::list< std::tuple< Domains::Point_vector, Domains::Point_vector > >& Match_wm_gm )
{
  //
  // k nearest neighbor data structure
  Tree tree;
  // build the tree
  for( auto gm_vertex : Gray_matter_surface_point_normal )
    tree.insert( gm_vertex );
 
  //
  //
  for ( auto wm_vertex : White_matter_surface_point_normal  )
    {
      //
      //
      Neighbor_search NN( tree, wm_vertex, 15 );
      auto filter_nearest = NN.begin();

      //
      // theta between white matter normal and gray matter normal is less than 30°
      float cos_theta = wm_vertex.cosine_theta( filter_nearest->first );
      //
      while( cos_theta < 0.87 && ++filter_nearest != NN.end() )
	cos_theta = wm_vertex.cosine_theta( filter_nearest->first );

      //
      // make tuple list
      if( filter_nearest != NN.end() )
	Match_wm_gm.push_back( std::make_tuple(wm_vertex, filter_nearest->first) );
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
