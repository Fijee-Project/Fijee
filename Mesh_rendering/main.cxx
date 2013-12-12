#include <iostream>
#include <thread>
//
// UCSF
//
#include "Access_parameters.h"
#include "Mesh_generator.h"
#include "Head_labeled_domain.h"
#include "Spheres_labeled_domain.h"
#include "Build_mesh.h"
#include "Conductivity_tensor.h"
#include "Utils/enum.h"
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
  parameters->init();
  
  // 
  // Head simulation:    Domains::Head_labeled_domain, Domains::Conductivity_tensor
  // Spheres simulation: Domains::Spheres_labeled_domain, Domains::Conductivity_tensor
  Domains::Mesh_generator< Domains::Head_labeled_domain, 
			   Domains::Conductivity_tensor > generator;
  //
  generator.make_inrimage();
  generator.make_conductivity();

//  // 
//  // Create the INRIMAGE
//  Domains::Head_labeled_domain labeled_image;
//  // Head segmentation
//  timerLog->MarkEvent("Head segmentation");
//  labeled_image.Head_model_segmentation();
//  // Write Inrimage
//  timerLog->MarkEvent("Write Inrimage");
//  labeled_image.Write_inrimage_file();



  // match white matter vertices with gray matter vertices
  timerLog->MarkEvent("match white matter vertices with gray matter vertices");
  parameters->epitaxy_growth();

//  //
//  // Diffusion tensor
//  timerLog->MarkEvent("Diffusion tensor");
//  Domains::Conductivity_tensor tensor;
//  //
//#ifdef TRACE
//#if ( TRACE == 200 )
//  tensor.VTK_visualization();
//#endif
//#endif
//  //
//  tensor.Move_conductivity_array_to_parameters();
//



  //
  // Tetrahedrization
  timerLog->MarkEvent("Build the mesh");
  Domains::Build_mesh tetrahedrization;
  //
  // Match conductivity with mesh's cells
  timerLog->MarkEvent("Mesh conductivity matching");
  tetrahedrization.Conductivity_matching();
  //
  // Build electrical dipoles list
  timerLog->MarkEvent("Build electrical dipoles list");
  tetrahedrization.Create_dipoles_list();

  //
  // Output
  timerLog->MarkEvent("write Outputs");
  //
#ifdef DEBUG
  // DEBUG MODE
  // Sequencing
  timerLog->MarkEvent("write Medit mesh");
  tetrahedrization.Output_mesh_format();
  timerLog->MarkEvent("write FEniCS mesh");
  tetrahedrization.Output_FEniCS_xml();
  timerLog->MarkEvent("write mesh conductivity");
  tetrahedrization.Output_mesh_conductivity_xml();
  timerLog->MarkEvent("write dipoles list");
  tetrahedrization.Output_dipoles_list_xml();
  //#ifdef TRACE
  //#if ( TRACE == 200 )
  //  tetrahedrization.Output_VTU_xml();
  //#endif
  //#endif
  //
#else
  // NO DEBUG MODE
  // Multi-threading
  std::thread output(std::ref(tetrahedrization), MESH_OUTPUT);
  std::thread subdomains(std::ref(tetrahedrization), MESH_SUBDOMAINS);
  std::thread conductivity(std::ref(tetrahedrization), MESH_CONDUCTIVITY);
  std::thread dipoles(std::ref(tetrahedrization), MESH_DIPOLES);
  //
  //#ifdef TRACE
  //#if ( TRACE == 200 )
  //  std::thread vtu(std::ref(tetrahedrization), MESH_VTU);
  //  vtu.join();
  //#endif
  //#endif
  //
  output.join();
  subdomains.join();
  conductivity.join();
  dipoles.join();
#endif

  //
  // Time log 
  timerLog->MarkEvent("Stop the process");
  std::cout << "Events log:" << *timerLog << std::endl;
 
  //
  //
  return EXIT_SUCCESS;
}
