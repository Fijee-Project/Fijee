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
//
// Project
//
#include "VTK_implicite_domain.h"
//
// CGAL
//
//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
//typedef CGAL::Point_with_normal_3<Kernel> Point_with_normal;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef CGAL::Poisson_reconstruction_function<Kernel> Poisson_reconstruction_function;
typedef CGAL::Implicit_surface_3<Kernel, Poisson_reconstruction_function> Surface_3;
//
// Eigen
//
#include <Eigen/Dense>
//
// VTK
//
typedef Domains::VTK_implicite_domain Domain;
//
//
//
Domain::VTK_implicite_domain():
  vtk_mesh_("")
{
  poly_data_bounds_[0] = 0.;
  poly_data_bounds_[1] = 0.;
  poly_data_bounds_[2] = 0.;
  poly_data_bounds_[3] = 0.;
  poly_data_bounds_[4] = 0.;
  poly_data_bounds_[5] = 0.;
}
//
//
//
Domain::VTK_implicite_domain( const char* Vtk_Mesh ):
  vtk_mesh_( Vtk_Mesh )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("VTK_implicite_domain constructor");

  //
  // create the stream for CGAL applications
  std::stringstream stream;

  //
  //
  vtkSmartPointer< vtkPoints > Points   = vtkSmartPointer< vtkPoints >::New();
  vtkSmartPointer< vtkPoints > Normals  = vtkSmartPointer< vtkPoints >::New();

  std::string line;
  float x[3]; // position
  float n[3]; // normal
  vtkIdType points_id[3]; // pts[3]
  vtkIdType normal_id;
  std::map< std::string, vtkIdType > Vertex_type_id;
  std::map< vtkIdType, std::set<vtkIdType> > Point_and_normals;

  //
  // Stereo Lithographie
  ifstream Stl_file ( Vtk_Mesh );
  if (Stl_file.is_open())
    {
      //
      //  Ingest header and junk to get to first vertex
      if ( ! getline (Stl_file,line) )
	{
	  //	  vtkErrorMacro ("STLReader error reading file: " << this->FileName
	  //			 << " Premature EOF while reading header.");
	  exit(-1);
	}
      
      //
      // main loop
      while ( Stl_file.good() )
	{
	  // facet normal
	  if ( getline (Stl_file,line) )
	    {
	      if( sscanf( line.c_str(), "%*s %*s %f %f %f\n", n, n+1, n+2 ) != 3 )
		{
		  break;
		}
	      //
	      normal_id = Normals->InsertNextPoint(n);
	    }
	  // outer loop
	  if ( getline (Stl_file,line) )
	    {
	    }
	  
	  //
	  // Vertices construction
	  // vertex 0
	  if ( getline (Stl_file,line) )
	    {
	      if (sscanf (line.c_str(), "%*s %f %f %f\n", x,x+1,x+2) != 3)
		{
		}
	      //
	      if( Vertex_type_id.find(line) == Vertex_type_id.end() )
		{
		  points_id[0] = Points->InsertNextPoint(x);
		  Vertex_type_id[line] = points_id[0];
		  Point_and_normals[ points_id[0] ].insert( normal_id );
		}
	      else
		Point_and_normals[ Vertex_type_id[line] ].insert( normal_id );
		
	    }
	  // vertex 1
	  if ( getline (Stl_file,line) )
	    {
	      if (sscanf (line.c_str(), "%*s %f %f %f\n", x,x+1,x+2) != 3)
		{
		}
	      //
	      if( Vertex_type_id.find(line) == Vertex_type_id.end() )
		{
		  points_id[1] = Points->InsertNextPoint(x);	    
		  Vertex_type_id[line] = points_id[1];
		  Point_and_normals[ points_id[1] ].insert( normal_id );
		}
	      else
		Point_and_normals[ Vertex_type_id[line] ].insert( normal_id );
	    }
	  // vertex 2
	  if ( getline (Stl_file,line) )
	    {
	      if (sscanf (line.c_str(), "%*s %f %f %f\n", x,x+1,x+2) != 3)
		{
		}
	      //
	      if( Vertex_type_id.find(line) == Vertex_type_id.end() )
		{
		  points_id[2] = Points->InsertNextPoint(x);	    
		  Vertex_type_id[line] = points_id[2];
		  Point_and_normals[ points_id[2] ].insert( normal_id );
		}
	      else
		Point_and_normals[ Vertex_type_id[line] ].insert( normal_id );
	    }

	  //
	  // endloop
	  if ( getline (Stl_file,line) )
	    {
	    }
	  // endfacet
	  if ( getline (Stl_file,line) )
	    {
	    }
	}
      //
      Stl_file.close();
    }
  else 
    {
      std::cout << "Unable to open file: " << Vtk_Mesh << std::endl;
      exit(-1);
    }

  //
  // MNI 305 coordinates
  Eigen::Matrix3f R_mni305;
  R_mni305 <<
    -1, 0, 0,
     0, 0, 1,
     0,-1, 0;
  //
  Eigen::Vector3f T_mni305;
  T_mni305 <<
     128,
    -128,
     128;
  
  //
  // We check how different we are from MNI 305 coordinates
  Eigen::Matrix3f aseg_rotation = 
    (Domains::Access_parameters::get_instance())->get_rotation_();
  Eigen::Vector3f aseg_translation = 
    (Domains::Access_parameters::get_instance())->get_translation_();
  //
  Eigen::Vector3f delta_traslation = aseg_translation - T_mni305;
  //
  Eigen::Matrix3f delta_rotation   = aseg_rotation - R_mni305;
  //
  // We don't have explicite example to be sure about the 
  // correct rotation to apply. All the next commented code
  // lines can help if we face this case.
  if ( delta_rotation.cwiseAbs().maxCoeff() > 1.e-03 )
    {
      std::cerr << delta_rotation << std::endl;
      std::cerr << "Patient position was very tilted compared to the MNI 305 coordinates" << std::endl;
      std::cerr << "We might need to implement a rotation like:" << std::endl;
      std::cerr << "R = Id + delta_rotation" << std::endl;
      exit( 1 );
    }
  // Set the delta transformation
  (Domains::Access_parameters::get_instance())->set_delta_rotation_(delta_rotation);
  (Domains::Access_parameters::get_instance())->set_delta_translation_(delta_traslation);


  //
  // Translate and rotate points with theire normal
  // Create a polydata object
  vtkSmartPointer<vtkPolyData> Poly_data_points =
    vtkSmartPointer<vtkPolyData>::New();
  Poly_data_points->SetPoints( Points );
  vtkSmartPointer<vtkPolyData> Poly_data_normals =
    vtkSmartPointer<vtkPolyData>::New();
  Poly_data_normals->SetPoints( Normals );
  //  // Transform the VTK mesh
  //  // bug Freesurfer: mirror symmetry
  //  double symmetry_matrix[16] = { 1, 0, 0, 0,
  //				   0, 1, 0, 38,
  //				   0, 0, 1, 6.,
  //				   0, 0, 0, 1};
  //  
  //  vtkSmartPointer<vtkTransform> symmetry    = vtkSmartPointer<vtkTransform>::New();
  //  symmetry->SetMatrix( symmetry_matrix );
  //  // rotation
  //  vtkSmartPointer<vtkTransform> rotation    = vtkSmartPointer<vtkTransform>::New();
  //  rotation->RotateWXYZ(90, 1, 0, 0);
  // translation
  vtkSmartPointer<vtkTransform> translation = vtkSmartPointer<vtkTransform>::New();
  //YC  translation->Translate(.0, 38., 6.);
  translation->Translate( delta_traslation(0), 
			  delta_traslation(1), 
			  delta_traslation(2) );
  //  // Points symmetry
  //  vtkSmartPointer<vtkTransformPolyDataFilter> transform_symmetric = 
  //    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  //  transform_symmetric->SetTransform( symmetry );
  //#if VTK_MAJOR_VERSION <= 5
  //  transform_symmetric->SetInputConnection( Poly_data_points->GetProducerPort() );
  //#else
  //  transform_symmetric->SetInputData( Poly_data_points );
  //#endif
  //  transform_symmetric->Update();
  //  // Points rotation
  //  vtkSmartPointer<vtkTransformPolyDataFilter> transform_rotation = 
  //    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  //  transform_rotation->SetTransform( rotation );
  //#if VTK_MAJOR_VERSION <= 5
  //  transform_rotation->SetInputConnection( transform_symmetric->GetOutputPort() );
  //#else
  //  transform_rotation->SetInputData( transform_symmetric );
  //#endif
  //  transform_rotation->Update();
  // Point translation
  vtkSmartPointer<vtkTransformPolyDataFilter> transform_translation = 
    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  transform_translation->SetTransform( translation );
#if VTK_MAJOR_VERSION <= 5
  transform_translation->SetInputConnection( Poly_data_points->GetProducerPort() );
#else
  transform_translation->SetInputData( Poly_data_points );
#endif
  transform_translation->Update();
  //  // Normals symmetry
  //  vtkSmartPointer<vtkTransformPolyDataFilter> normals_symmetric = 
  //    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  //  normals_symmetric->SetTransform( symmetry );
  //#if VTK_MAJOR_VERSION <= 5
  //  normals_symmetric->SetInputConnection( Poly_data_normals->GetProducerPort() );
  //#else
  //  normals_symmetric->SetInputData( Poly_data_normals );
  //#endif
  //  normals_symmetric->Update();
  //  // Normals rotation
  //  vtkSmartPointer<vtkTransformPolyDataFilter> normals_rotation = 
  //    vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  //  normals_rotation->SetTransform( rotation );
  //#if VTK_MAJOR_VERSION <= 5
  //  normals_rotation->SetInputConnection( normals_symmetric->GetOutputPort() );
  //#else
  //  normals_rotation->SetInputData( normals_symmetric );
  //#endif
  //  normals_rotation->Update();

  //
  // Record the boundery data
  Poly_data_points->GetBounds(poly_data_bounds_);
 
  //
  // Create the Point with normal contenair
  double vertex[3];
  double 
    normal[3],
    normal_part[3];
  double norm;
  //
  std::map< vtkIdType, std::set< vtkIdType > >::iterator it;
  for ( it  = Point_and_normals.begin() ; 
	it != Point_and_normals.end() ; 
	++it )
    {
      //      Points->GetPoint( it->first, vertex);
      //     Poly_data_points->GetPoint( it->first, vertex);
      transform_translation->GetOutput()->GetPoint( it->first, vertex );
      //      transform_symmetric->GetOutput()->GetPoint( it->first, vertex );
      //      std::cout << vertex[0] << " " << vertex[1] << " " << vertex[2] << " ";
      //
      normal[0] = 0.;
      normal[1] = 0.;
      normal[2] = 0.;
      std::set<vtkIdType>::iterator it_normal;
      for ( it_normal  = (it->second).begin(); 
	    it_normal != (it->second).end(); 
	    ++it_normal)
	{
	  //	  normal = 
	  //	  Normals->GetPoint( (*it_normal), normal_part);
	  Poly_data_normals->GetPoint( (*it_normal), normal_part);
	  //	  normals_rotation->GetOutput()->GetPoint( (*it_normal), normal_part);
	  //	  normals_symmetric->GetOutput()->GetPoint( (*it_normal), normal_part);
	  normal[0] += normal_part[0];
	  normal[1] += normal_part[1];
	  normal[2] += normal_part[2];
	}
      norm = sqrt( normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2] );
      normal[0] /= norm;
      normal[1] /= norm;
      normal[2] /= norm;
      //      std::cout << normal[0] << " " << normal[1] << " " << normal[2] << " " << std::endl;
      //
      point_normal_.push_back(Domains::Point_vector( vertex[0], vertex[1], vertex[2],
						     normal[0], normal[1], normal[2] ));
      //
      stream << vertex[0] << " " << vertex[1] << " " << vertex[2] << " " 
	     << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;
    }

  //
  //
  std::vector<Point_with_normal> Point_normal;
  if (!stream ||
      !CGAL::read_xyz_points_and_normals( stream,
					  std::back_inserter(Point_normal),
					  CGAL::make_normal_of_point_with_normal_pmap( std::vector<Point_with_normal>::value_type() )))
    {
      std::cerr << "Reading error with the stream of point with normal" << std::endl;
      exit(-1);
    }

  //
  // Creates implicit function from the read points using the default solver.
  function_.reset( new Poisson_reconstruction_function( Point_normal.begin(), 
							Point_normal.end(),
							CGAL::make_normal_of_point_with_normal_pmap( std::vector<Point_with_normal>::value_type() ) ) );
}
//
//
//
void
Domain::operator ()( double** Space_Points )
{
  // Computes the Poisson indicator function f()
  // at each vertex of the triangulation.
  // smoother_hole_filling = true: controls if the Delaunay refinement is done for the input 
  // points
  if ( !function_->compute_implicit_function() )
    exit(-1);
  // Check if the surface is inside out
  if( inside_domain( CGAL::Point_3< Kernel > (256., 256., 256.) ) )
    function_->flip_f();
}
//
//
//
Domain::VTK_implicite_domain( const Domain& that )
{
}
//
//
//
Domain::VTK_implicite_domain( Domain&& that )
{
}
//
//
//
Domain& 
Domain::operator = ( const Domain& that )
{
  if ( this != &that ) 
    {
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
    }
  //
  return *this;
}
//
//
//
Domain& 
Domain::operator = ( Domain&& that )
{
  if( this != &that )
    {
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
    }
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const Domain& that)
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
