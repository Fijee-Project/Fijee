set(UFL_SOURCES SLS_model.ufl SLD_model.ufl tCS_model.ufl tCS_current_density_model.ufl tCS_electrical_field_model.ufl)
set(UFL_FLAG -l dolfin)
set(UFL_COMPILER ffc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(UFL_HEADERS)
foreach(_file ${UFL_SOURCES}) 
  string(REPLACE ".ufl" ".h" ufl_header ${_file})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_header}
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ffc ${UFL_FLAG}  ${CMAKE_CURRENT_SOURCE_DIR}/${_file} 
    COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}
  )
  list(APPEND UFL_HEADERS ${ufl_header})
endforeach()
add_custom_target(ufl_headers DEPENDS ${UFL_HEADERS})



include_directories(${CMAKE_CURRENT_BINARY_DIR})
  



set(POISSON_SOURCES main.cxx  Conductivity.cxx Source.cxx Electrodes_injection.cxx Intensity.cxx Physics.cxx SL_subtraction.cxx SL_direct.cxx tCS_tDCS.cxx PDE_solver_parameters.cxx Spheres_electric_monopole.cxx Electrodes_setup.cxx Electrodes_surface.cxx) #Spheres_electric_dipole.cxx

add_executable(poisson ${POISSON_SOURCES})
add_dependencies(poisson ufl_headers)

target_link_libraries(poisson ${LIBS} pugi)

message(STATUS ${LIBS})



