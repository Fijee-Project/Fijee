# Barebones CMAKE script for fijee project
project( Fijee CXX C ) 

#
#
#include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

#
# Minimal version of CMake:
cmake_minimum_required( VERSION 2.6.0 )

#
# Some best practice versioning and soversioning for now
set( Fijee_NAME ${PROJECT_NAME} )
set( Fijee_VERSION_MAJOR "0" )
set( Fijee_VERSION_MINOR "1" )
set( Fijee_VERSION_PATCH "1" )
#
set(Fijee_VERSION "${Fijee_VERSION_MAJOR}.${Fijee_VERSION_MINOR}.${Fijee_VERSION_PATCH}")
#
set( LIBRARY_VERSION "0.0.1" )
set( LIBRARY_SOVERSION "0" )


#
#
set( Fijee_CXX_DEFINITIONS "" )
set( Fijee_LINK_FLAGS "" )
set( Fijee_Utils_TARGET_LINK_LIBRARIES "" )
set( Fijee_Biophysics_TARGET_LINK_LIBRARIES "" )
set( Fijee_Mesh_TARGET_LINK_LIBRARIES "" )
set( Fijee_FEM_TARGET_LINK_LIBRARIES "" )
# DOLFIN INCOMPATIBILITY #
# DOLFIN INCOMPATIBILITY # TRACE:
# DOLFIN INCOMPATIBILITY # 1, 2, 3, ... printout
# DOLFIN INCOMPATIBILITY # 100 R statistical software studies
# DOLFIN INCOMPATIBILITY # 200 VTK visualization
# DOLFIN INCOMPATIBILITY if( NOT DEFINED Fijee_TRACE )
# DOLFIN INCOMPATIBILITY     set( Fijee_TRACE 0 )
# DOLFIN INCOMPATIBILITY endif()
# DOLFIN INCOMPATIBILITY #
# DOLFIN INCOMPATIBILITY add_definitions( -DTRACE=${Fijee_TRACE} )
# DOLFIN INCOMPATIBILITY #
# DOLFIN INCOMPATIBILITY # CUDA:
# DOLFIN INCOMPATIBILITY if( NOT DEFINED Fijee_CUDA )
# DOLFIN INCOMPATIBILITY     set( Fijee_CUDA "OFF" )
# DOLFIN INCOMPATIBILITY else()
# DOLFIN INCOMPATIBILITY   if(Fijee_CUDA)
# DOLFIN INCOMPATIBILITY     add_definitions( -DCUDA )
# DOLFIN INCOMPATIBILITY   endif()
# DOLFIN INCOMPATIBILITY endif()
#
#
add_definitions( -std=c++0x -Wno-deprecated -DVERSION_OF_FIJEE="${Fijee_VERSION}" )  

#
# Setup files
#

#
# Fijee data set
set( Fijee_DATA_SET ${Fijee_SOURCE_DIR}/share/fijee.xml )
#
install( FILES ${Fijee_DATA_SET} DESTINATION ${CMAKE_INSTALL_PREFIX}/share )

#
# Electrodes
set( Fijee_ELECTRODES_STANDARD_10_20_CAP81 ${Fijee_SOURCE_DIR}/share/electrodes/electrodes-Standard-10-20-Cap81.xml )
#
install( FILES ${Fijee_ELECTRODES_STANDARD_10_20_CAP81} DESTINATION ${CMAKE_INSTALL_PREFIX}/share/electrodes/ )


#
# Find required packages
#
message("\n== Find required packages ==\n")


# Tell cmake where to look for modules, modules are scripts that help us find required libraries
set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/cmake/modules;${CMAKE_MODULE_PATH}"
)


#
# General libraries
find_package( PkgConfig REQUIRED)
#pkg_check_modules( metis REQUIRED metis )


#
# FEniCS
find_package(DOLFIN REQUIRED CONFIG)
if(DOLFIN_FOUND)
  message( STATUS "FEniCS found at ${DOLFIN_INCLUDE_DIRS}" )
  include(${DOLFIN_USE_FILE})
  get_filename_component( DOLFIN_LIBRARY_DIRS ${DOLFIN_LIBRARIES} PATH )
endif()

#
# VTK
find_package(VTK 5.10 REQUIRED NO_MODULE)
if(VTK_FOUND)
  message(STATUS "VTK found")
  include(${VTK_USE_FILE})
endif()

#
# CUDA
if( Fijee_CUDA )
  find_package( CUDA REQUIRED )
  if(CUDA_FOUND)
    set( CUDA_NVCC_FLAGS "-O3 -gencode arch=compute_20,code=sm_20" )
  endif()
endif()


##
## BOOST
#find_package(Boost COMPONENTS mpi program_options filesystem thread system iostreams)
#if(Boost_FOUND)
#  include_directories(${Boost_INCLUDE_DIRS})
#  list(APPEND LIBS ${Boost_LIBRARIES})
#endif()
##
#add_definitions( -DBOOST_ALL_DYN_LINK)

#
# Eigen
find_package(EIGEN 3.0 REQUIRED)
#
if(EIGEN_FOUND)
  message(STATUS "eigen found @ ${EIGEN_DIR} ${EIGEN_INCLUDE_DIR}")
  include_directories(${EIGEN_INCLUDE_DIR})
endif()

##
## MPI (FEniCS)
#find_package(MPI REQUIRED)
##
#if( MPI_FOUND )
#  message(STATUS "MPI found" )# @ ${MPI_INCLUDE_PATH} : ")
#  include_directories(${MPI_INCLUDE_PATH})
#endif()


#
# Metis
find_package( METIS REQUIRED )
#
if( METIS_FOUND )
  message( STATUS "Metis found" )
  include_directories( ${METIS_INCLUDE_DIR} )
endif()

#
# NIFTI
find_package(NIFTI REQUIRED)
#
if(NIFTI_FOUND)
  message(STATUS "nifti found")
  include_directories(${NIFTI_INCLUDE_DIR})
endif()

#
# CGAL
find_package(CGAL 4.3 REQUIRED)
#
if(CGAL_FOUND)
  message( STATUS "CGAL found ${CGAL_DIR}" )
  add_definitions( -DCGAL_EIGEN3_ENABLED )
  include( ${CGAL_USE_FILE} )
endif()

#
include_directories( SYSTEM )

#
# Build 
#
message( "\n== Generating build files ==\n" )

#
# Env
set(Fijee_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(Fijee_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(Fijee_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
set(Fijee_CMAKE_DIR ${CMAKE_INSTALL_PREFIX}/lib/cmake/Fijee-${Fijee_VERSION})

#
# Import all the source 
add_subdirectory(Fijee)

#
# Fijee configuration
#
message( "\n== Configuration files ==\n" )

# Add all targets to the build-tree export set
export( TARGETS FijeeUtils FijeeBiophysics FijeeMesh
  FILE "${PROJECT_BINARY_DIR}/FijeeTargets.cmake" )
#
include_directories( "${Fijee_INCLUDE_DIR}" )
# Create the FijeeConfig.cmake and FijeeConfigVersion files
file( RELATIVE_PATH REL_INCLUDE_DIR 
  "${Fijee_CMAKE_DIR}" 
  "${Fijee_INCLUDE_DIR}" )
# ... for the build tree
set( CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" )
configure_file( FijeeConfig.cmake.in "${PROJECT_BINARY_DIR}/FijeeConfig.cmake" @ONLY )
# ... for the install tree
set( CONF_INCLUDE_DIRS "\${Fijee_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
configure_file( FijeeConfig.cmake.in 
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FijeeConfig.cmake" @ONLY )
# ... for both
configure_file( FijeeConfigVersion.cmake.in 
  "${PROJECT_BINARY_DIR}/FijeeConfigVersion.cmake" @ONLY)
# Install the FijeeConfig.cmake and FijeeConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FijeeConfig.cmake"
  "${PROJECT_BINARY_DIR}/FijeeConfigVersion.cmake"
  DESTINATION "${Fijee_CMAKE_DIR}" COMPONENT dev)
# Install the export set for use with the install-tree
install(EXPORT FijeeTargets DESTINATION
  "${Fijee_CMAKE_DIR}" COMPONENT dev)

