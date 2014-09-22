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
#ifndef FIJEE_FEM_H
#define FIJEE_FEM_H
/*!
 * \file Fijee_fem.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <Finite_element_method_models/PDE_solver_parameters.h>
// Geometry and physics
#include <Finite_element_method_models/Source.h>
#include <Finite_element_method_models/Intensity.h>
#include <Finite_element_method_models/Field.h>
#include <Finite_element_method_models/Boundaries.h>
#include <Finite_element_method_models/Conductivity.h>
#include <Finite_element_method_models/Sub_domaines.h>
#include <Finite_element_method_models/Parcellation_information.h>
#include <Finite_element_method_models/Spheres_electric_monopole.h>
// Source localization and tCS
#include <Finite_element_method_models/Model_solver.h>
#include <Finite_element_method_models/Physics.h>
#include <Finite_element_method_models/SL_subtraction.h>
#include <Finite_element_method_models/SL_direct.h>
#include <Finite_element_method_models/tCS_tDCS.h>
#include <Finite_element_method_models/tCS_tACS.h>
#include <Finite_element_method_models/tCS_tDCS_local_conductivity.h>
// Electrodes
#include <Finite_element_method_models/Electrodes_setup.h>
#include <Finite_element_method_models/Electrodes_surface.h>
#include <Finite_element_method_models/Electrodes_injection.h>
// UFL
#endif
