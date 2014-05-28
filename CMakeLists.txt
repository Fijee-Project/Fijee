### Barebones CMAKE script for fijee project



project (FIJEE CXX) 

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cmake_minimum_required(VERSION 2.6.0)

#Some best practice versioning and soversioning for now
set(APPLICATION_NAME ${PROJECT_NAME})
set(APPLICATION_VERSION_MAJOR "0")
set(APPLICATION_VERSION_MINOR "1")
set(APPLICATION_VERSION_PATCH "1")

set(APPLICATION_VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}.${APPLICATION_VERSION_PATCH}")

set(LIBRARY_VERSION "0.0.1")
set(LIBRARY_SOVERSION "0")
##


##This needs to be changed to use cmake scripts to determine correct flag
set(FIJEE_CXX_FLAGS "-std=c++0x")
list(APPEND CMAKE_CXX_FLAGS ${FIJEE_CXX_FLAGS})



#Tell cmake where to look for modules, modules are scripts that help us find required libraries

set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/cmake/modules;${CMAKE_MODULE_PATH}"
)

#include(FijeeDefaults)
set(LIBS)

set(DOLFIN_DIR "/FIJEE-AUX/share/dolfin/cmake") # Dolfin installs UseDolfin.cmake scripts, dummy findDolfin script in cmake/modules doesn't look here properly yet

find_package(DOLFIN REQUIRED)
if(DOLFIN_FOUND)
  include_directories(${DOLFIN_INCLUDE_DIRS})
  list (APPEND LIBS ${DOLFIN_LIBRARIES})
endif()


#set(VTK_DIR "/FIJEE-AUX/lib/vtk-5.10/")
find_package(VTK REQUIRED NO_MODULE)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
  list(APPEND LIBS ${VTK_LIBRARIES})
endif()
#link_directories(${VTK_LIBRARIES})

find_package(Boost COMPONENTS mpi program_options filesystem thread system iostreams)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND LIBS ${Boost_LIBRARIES})
endif()

#link_directories(${Boost_LIBRARIES})

add_definitions( -DBOOST_ALL_DYN_LINK)

find_package(EIGEN 3.0 REQUIRED)
if(EIGEN_FOUND)
  message(STATUS "eigen found @ ${EIGEN_DIR} ${EIGEN_INCLUDE_DIR}")
  include_directories(${EIGEN_INCLUDE_DIR})
endif()
####


#import all the source folder
add_subdirectory(Finite_element_method_models)
add_subdirectory(Utils)
#add_subdirectory(Mesh_rendering) # Mesh rendering not done yet


