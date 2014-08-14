###############
## CONFIGURE ##
###############
export PREFIX =
export PATH_SOFT =/home/cobigo/devel/CPP/
DEBUG         = no
VERSION       = 1.0
DIST          = Fijee

#################
## COMPILATION ##
#################
export CXX  = g++
export CC   = g++
export CUDA = nvcc
#-lineinfo CUDA
#  -gencode arch=compute_10,code=sm_10 
#  -gencode arch=compute_20,code=sm_20 
#  -gencode arch=compute_30,code=sm_30 
#  -gencode arch=compute_35,code=sm_35 
#
ifeq ($(DEBUG),yes)
CXXFLAGS_MODE  = -g -O0 -DDEBUG
CUDAFLAGS_MODE = 
else
CXXFLAGS_MODE  = -O3 -Wunused-local-typedefs #-DGPU
CUDAFLAGS_MODE = -O3 -m64 -gencode arch=compute_20,code=sm_20 
endif
#
export CXXFLAGS  += $(CXXFLAGS_MODE) -Wno-deprecated -std=c++11 -frounding-math -DCGAL_EIGEN3_ENABLED -DDEBUG_UCSF #-DDEBUG_TRACE -std=c++0x -std=c++11 or -std=gnu++11
export CUDAFLAGS += $(CUDAFLAGS_MODE)
export UFL = ffc

# Warning: -Wno-deprecate might cause difficult linking issue difficult to solve

####################
## TIER LIBRARIES ##
####################
CGAL_DIR  = $(PATH_SOFT)/CGAL/
NIFTI_DIR = $(PATH_SOFT)/nifti
#
# VTK
#
VTK_DIR     = $(PATH_SOFT)/VTK
export VTK_VERSION = vtk-5.10
export VTK_LIBS    = -lvtkCommon -lvtkFiltering -lvtkGenericFiltering -lvtkGraphics -lvtkIO -lvtkRendering -lvtksys -lvtkVolumeRendering -lvtkzlib -lvtkfreetype -lvtkftgl -lvtkImaging -lvtkhdf5 -lvtkhdf5_hl -lvtkexpat -lvtktiff -lvtkjpeg -lvtkpng -lvtksqlite -lvtkmetaio -lLSDyna -lvtkNetCDF -lvtkDICOMParser -lvtkverdict -lvtkNetCDF -lvtkHybrid -lvtkNetCDF_cxx -lvtkexoIIc -lvtklibxml2 -lvtkalglib -lvtkproj4  
#
#
#
ifeq ($(DEBUG),yes)
export CGAL   = $(CGAL_DIR)/install
export NIFTI  = $(NIFTI_DIR)/install
export VTK    = $(VTK_DIR)/install
else
export CGAL   = $(CGAL_DIR)/install
export NIFTI  = $(NIFTI_DIR)/install
export VTK    = $(VTK_DIR)/install
endif
#
export EIGEN3   = $(PATH_SOFT)/Eigen/install
export FENICS   = $(PATH_SOFT)/FEniCS/install/
export CUDA_LIB = /usr/local/cuda/lib64/

######################
## Main DIRECTORIES ##
######################
export FIJEE += $(CURDIR)

#####################
## SUB DIRECTORIES ##
#####################
UTILS_DIR              = Utils
MESH_RENDERING_DIR     = Mesh_rendering
FEM_MODELS_DIR         = Finite_element_method_models
ELECTRODES_DIR         = Electrodes
EXEC = $(MESH_RENDERING)/build_inrimage  $(FEM_MODELS_DIR)/Poisson

###############
## EXECUTION ##
###############
all: models $(EXEC)

$(EXEC):
	( cd $(UTILS_DIR)/pugi/ && $(MAKE) )
	( cd $(UTILS_DIR)/Minimizers/ && $(MAKE) )
	( cd $(UTILS_DIR)/Biophysics/ && $(MAKE) )
	( cd $(FEM_MODELS_DIR) && $(MAKE) )
	( cd $(MESH_RENDERING_DIR) && $(MAKE) )
	( cd $(ELECTRODES_DIR) && $(MAKE) )
	@echo""
	@echo "export LD_LIBRARY_PATH=$(FIJEE)/$(UTILS_DIR)/pugi/:$(FIJEE)/$(UTILS_DIR)/Minimizers/:$(FIJEE)/$(UTILS_DIR)/Biophysics/:$(CUDA_LIB):$(VTK)/lib/vtk-5.10:$(CGAL)/lib:$(LD_LIBRARY_PATH)"
	@echo""


models:
	( cd $(FEM_MODELS_DIR) && $(MAKE) models )


clean:
	( cd $(UTILS_DIR)/pugi/ && $(MAKE) $@ )
	( cd $(UTILS_DIR)/Minimizers/ && $(MAKE) $@ )
	( cd $(UTILS_DIR)/Biophysics/ && $(MAKE) $@ )
	( cd $(FEM_MODELS_DIR) && $(MAKE) $@ )
	( cd $(MESH_RENDERING_DIR) && $(MAKE) $@ )
	( cd $(ELECTRODES_DIR) && $(MAKE) $@ )

distclean: clean
	find . -name *~      -exec rm {} \;
	find . -name SLS_model.h -exec rm {} \;
	find . -name SLD_model.h -exec rm {} \;
	find . -name tCS_model.h -exec rm {} \;
	find . -name tCS_current_density_model.h -exec rm {} \;
	find . -name tCS_electrical_field_model.h -exec rm {} \;
	find . -name *.mesh  -exec rm {} \;
	find . -name *.inr   -exec rm {} \;
	find . -name *.frame -exec rm {} \;

#check:
#	
#
#install:
#	
#
dist:
	mkdir $(DIST)
	cp Makefile $(DIST)/
	mkdir $(DIST)/$(FEM_MODELS_DIR)
	mkdir $(DIST)/$(MESH_RENDERING_DIR)
	cp $(FEM_MODELS_DIR)/Makefile      $(DIST)/$(FEM_MODELS_DIR)/
	cp $(MESH_RENDERING_DIR)/{Makefile,README} $(DIST)/$(MESH_RENDERING_DIR)/      
	cp $(FEM_MODELS_DIR)/*.{h,cxx,ufl} $(DIST)/$(FEM_MODELS_DIR)/
	cp $(MESH_RENDERING_DIR)/*.{h,cxx,cu}      $(DIST)/$(MESH_RENDERING_DIR)/      
	tar zcvf $(DIST)-$(VERSION).tar.gz $(DIST)
	rm -rf $(DIST)
