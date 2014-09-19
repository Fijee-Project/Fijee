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
#ifndef FIJEE_MESH_RENDERING_H
#define FIJEE_MESH_RENDERING_H
/*!
 * \file Fijee_mesh_rendering.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <Mesh_rendering/Access_parameters.h>
//
#include <Mesh_rendering/Point.h>
#include <Mesh_rendering/Point_vector.h>
#include <Mesh_rendering/Distance.h>
#include <Mesh_rendering/Dipole.h>
#include <Mesh_rendering/Cell_conductivity.h>
#include <Mesh_rendering/CUDA_Conductivity_matching.h>
#include <Mesh_rendering/CUDA_Conductivity_matching_functions.h>
#include <Mesh_rendering/Build_dipoles_list.h>
#include <Mesh_rendering/Build_dipoles_list_knn.h>
#include <Mesh_rendering/Build_dipoles_list_high_density.h>
//
#include <Mesh_rendering/CGAL_tools.h>
//
#include <Mesh_rendering/Conductivity_tensor.h>
#include <Mesh_rendering/Head_conductivity_tensor.h>
#include <Mesh_rendering/Spheres_conductivity_tensor.h>
//
#include <Mesh_rendering/Electrode.h>
#include <Mesh_rendering/Electrode_shape.h>
#include <Mesh_rendering/Electrode_shape_sphere.h>
#include <Mesh_rendering/Electrode_shape_cylinder.h>
#include <Mesh_rendering/Build_electrodes_list.h>
//
#include <Mesh_rendering/Parcellation.h>
#include <Mesh_rendering/Parcellation_method.h>
#include <Mesh_rendering/Parcellation_METIS.h>
#include <Mesh_rendering/Parcellation_Scotch.h>
//
#include <Mesh_rendering/CGAL_image_filtering.h>
#include <Mesh_rendering/CGAL_implicite_domain.h>
#include <Mesh_rendering/VTK_implicite_domain.h>
#include <Mesh_rendering/Implicite_domain.h>
#include <Mesh_rendering/Spheres_implicite_domain.h>
#include <Mesh_rendering/Labeled_domain.h>
#include <Mesh_rendering/Head_labeled_domain.h>
#include <Mesh_rendering/Spheres_labeled_domain.h>
#include <Mesh_rendering/Mesh_generator.h>
#include <Mesh_rendering/Build_mesh.h>
#endif
