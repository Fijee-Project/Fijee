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
#include <iostream>
//
// UCSF
//
#include <fijee.h>
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
  FIJEE_TIME_PROFILER("main");

  //
  // TODO remove VTK traces
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
  // Head simulation:    
  //   - Domains::Head_labeled_domain 
  //   - Domains::Head_conductivity_tensor
  //   - Domains::Build_mesh
  //   - Dipole generation
  //     - Domains::Build_dipoles_list_high_density
  //     - Build_dipoles_list_knn
  //
  // Spheres simulation: 
  //   - Domains::Spheres_labeled_domain 
  //   - Domains::Spheres_conductivity_tensor
  //   - Domains::Build_mesh
  //   - Dipole generation
  //     - Domains::Build_dipoles_list_high_density
  //     - Build_dipoles_list_knn
  // 
  Domains::Mesh_generator< Domains::Head_labeled_domain, 
			   Domains::Head_conductivity_tensor,
			   Domains::Build_mesh,
			   Domains::Build_dipoles_list_high_density > generator;
  //
  generator.make_inrimage();
  generator.make_conductivity();
  // 
  generator.make_output();

//  // 
//  // Modelisation of alpha rhythm
//  //  - Utils::Biophysics::Jansen_Rit_1995
//  //  - Utils::Biophysics::Wendling_2002
//  //  - Utils::Biophysics::Molaee_Ardekani_Wendling_2009
//  // 
//  Utils::Biophysics::Brain_rhythm_models< Utils::Biophysics::Jansen_Rit_1995, 
//					  /*solver_parameters->get_number_of_threads_()*/ 4 > 
//    alpha;
//  //
//  alpha.modelization( parameters->get_files_path_output_() );
//  alpha.output();


  //
  // Time log 
  timerLog->MarkEvent("Stop the process");
  std::cout << "Events log:" << *timerLog << std::endl;
 
  //
  //
  return EXIT_SUCCESS;
}
