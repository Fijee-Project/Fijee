#include <iostream>
#include <fstream>      // std::ifstream, std::ofstream
#include <sstream>
#include <cstdlib>
#include <math.h> 
//
// VTK
//
#include <vtkVersion.h>
#include <vtkProperty2D.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkLabeledDataMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAxesActor.h>
#include <vtkTransform.h>
#include <vtkStringArray.h>
//
#define PI 3.14159265359

int main()
{
  //
  //
  // Create the geometry of a point (the coordinate)
  vtkSmartPointer<vtkPoints> points =
    vtkSmartPointer<vtkPoints>::New();
 
  // Create the topology of the point (a vertex)
  vtkSmartPointer<vtkCellArray> vertices =
    vtkSmartPointer<vtkCellArray>::New();

  //
  //
  std::ofstream file_[4];
  //
  std::stringstream output_stream_[4];
  //
  const char* file_name[4] = {"sphere_brain.xyz", "sphere_CSF.xyz", "sphere_skull.xyz", "sphere_scalp.xyz"};


  //
  // head dimensions
  float r[4] = {78., 80., 86., 92.};
  //            ^    ^    ^    ^
  //        brain  CSF  skull  scalp
  // number of segments
  int N[4] = {180, 200, 220, 240};
  //           ^    ^    ^    ^
  //       brain   CSF skull scalp
  // 
  float delta_theta[4];
  delta_theta[0] = PI / ((float) N[0]); /* brain */
  delta_theta[1] = PI / ((float) N[1]); /* CSF */
  delta_theta[2] = PI / ((float) N[2]); /* skull */
  delta_theta[3] = PI / ((float) N[3]); /* scalp */
  //
  float delta_phi[4];
  delta_phi[0] = 2 * PI / ((float) N[0]); /* brain */
  delta_phi[1] = 2 * PI / ((float) N[1]); /* CSF */
  delta_phi[2] = 2 * PI / ((float) N[2]); /* skull */
  delta_phi[3] = 2 * PI / ((float) N[3]); /* scalp */

  //
  vtkIdType points_id[ N[0] * N[0] + 1 ];
  int id = 0;


  //
  //
  for ( int seg = 0 ; seg < 4 ; seg++)
    {
      for ( int n_theta = 1 ; n_theta < N[seg] ; n_theta++ )
	for ( int n_phi = 1 ; n_phi < N[seg]   ; n_phi++ )
	  {
	    //
	    //
	    float x = r[seg] * 
	      cos(n_phi * delta_phi[seg]) * sin(n_theta * delta_theta[seg]);
	    float y = r[seg] * 
	      sin(n_phi * delta_phi[seg]) * sin(n_theta * delta_theta[seg]);
	    float z = r[seg] * cos(n_theta * delta_theta[seg]);
	    // normal
	    float x_normal = 
	      cos(n_phi * delta_phi[seg]) * sin(n_theta * delta_theta[seg]);
	    float y_normal = 
	      sin(n_phi * delta_phi[seg]) * sin(n_theta * delta_theta[seg]);
	    float z_normal = cos(n_theta * delta_theta[seg]);
	    
	    //
	    //
	    output_stream_[seg]
	      << x << " " << y << " " << z << " "
	      << x_normal << " " << y_normal << " " << z_normal << " "
	      << std::endl;
	    
	    if ( seg == 0 )
	      {
		float p[3] = {x, y, z};
		points_id[id] = points->InsertNextPoint(p);
		vertices->InsertNextCell(id++,points_id);
	      }
	  }

      //
      //
      file_[seg].open( file_name[seg] );
      file_[seg] << output_stream_[seg].rdbuf();
      file_[seg].close();  
    }

//  //
//  //  // Create a polydata object
//  vtkSmartPointer<vtkPolyData> point =
//    vtkSmartPointer<vtkPolyData>::New();
// 
//  // Set the points and vertices we created as the geometry and topology of the polydata
//  point->SetPoints( points );
//  point->SetVerts( vertices );
//
//  // Visualize
//  vtkSmartPointer<vtkPolyDataMapper> mapper =
//    vtkSmartPointer<vtkPolyDataMapper>::New();
//#if VTK_MAJOR_VERSION <= 5
//  mapper->SetInput(point);
//#else
//  mapper->SetInputData(point);
//#endif
// 
//  vtkSmartPointer<vtkActor> actor =
//    vtkSmartPointer<vtkActor>::New();
//  actor->SetMapper( mapper );
//  actor->GetProperty()->SetPointSize(2);
// 
//  vtkSmartPointer<vtkRenderer> renderer =
//    vtkSmartPointer<vtkRenderer>::New();
//  vtkSmartPointer<vtkRenderWindow> renderWindow =
//    vtkSmartPointer<vtkRenderWindow>::New();
//  renderWindow->AddRenderer(renderer);
//  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
//    vtkSmartPointer<vtkRenderWindowInteractor>::New();
//  renderWindowInteractor->SetRenderWindow(renderWindow);
// 
//  renderer->AddActor(actor);
// 
//  vtkSmartPointer<vtkTransform> transform =
//    vtkSmartPointer<vtkTransform>::New();
//  transform->Translate(2.0, 2.0, 0.0);
// 
//  vtkSmartPointer<vtkAxesActor> axes =
//    vtkSmartPointer<vtkAxesActor>::New();
//  axes->AxisLabelsOff();
//  // The axes are positioned with a user transform
//  axes->SetUserTransform(transform);
// 
//  renderer->AddActor(axes);
//
//  renderWindow->Render();
//  renderWindowInteractor->Start();


  //
  //
  return EXIT_SUCCESS;
}
