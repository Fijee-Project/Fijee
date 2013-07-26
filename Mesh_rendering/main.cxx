#include <iostream>
#include <thread>
//
// UCSF
//
#include "Access_parameters.h"
#include "Build_labeled_domain.h"
#include "Build_mesh.h"
#include "Conductivity_tensor.h"
//
// VTK
//
#include <vtkSmartPointer.h>
#include <vtkTimerLog.h>

//
// Name space
//

int 
main()
{
  //
  // Time log
  vtkSmartPointer<vtkTimerLog> timerLog = 
    vtkSmartPointer<vtkTimerLog>::New();
  //
  std::cout << "Process started at: " << timerLog->GetUniversalTime() << std::endl;

  // 
  // Access parameters
  Domains::Access_parameters* parameters = Domains::Access_parameters::get_instance();
  
  // 
  // Create the INRIMAGE
  Domains::Build_labeled_domain labeled_image;
  // Head segmentation
  timerLog->MarkEvent("Head segmentation");
  labeled_image.Head_model_segmentation();
  // Write Inrimage
  timerLog->MarkEvent("Write Inrimage");
  labeled_image.Write_inrimage_file();

  //
  // Diffusion tensor
  timerLog->MarkEvent("Diffusion tensor");
  Domains::Conductivity_tensor tensor;
//  tensor.VTK_visualization();

  //
  // Tetrahedrization
  timerLog->MarkEvent("Build the mesh");
  Domains::Build_mesh tetrahedrization;
  //
#ifdef DEBUG
  //
  timerLog->MarkEvent("write Outputs");
  //
  tetrahedrization.Output_mesh_format();
  //
  timerLog->MarkEvent("write FEniCS mesh");
  tetrahedrization.Output_FEniCS_xml();
//  tetrahedrization.Output_VTU_xml();
  //
  timerLog->MarkEvent("write mesh conductivity");
  tetrahedrization.Output_mesh_conductivity_xml();
#else
  std::thread output(std::ref(tetrahedrization), MESH_OUTPUT);
  std::thread subdomains(std::ref(tetrahedrization), MESH_SUBDOMAINS);
//  std::thread vtu(std::ref(tetrahedrization), MESH_VTU);
  std::thread conductivity(std::ref(tetrahedrization), MESH_CONDUCTIVITY);
  //
  output.join();
  subdomains.join();
//  vtu.join();
  conductivity.join();
#endif
  //
  // Time log 
  timerLog->MarkEvent("Stop the process");
  std::cout << "Events log:" << *timerLog << std::endl;
 
  //
  //
  return EXIT_SUCCESS;
}
