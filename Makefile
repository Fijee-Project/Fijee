###############
## CONFIGURE ##
###############
export PREFIX =
export PATH_SOFT =/home/cobigo/devel/CPP/
DEBUG         = no
VERSION       = 1.0
DIST          = fijee

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
CXXFLAGS_MODE  = -g -DDEBUG
CUDAFLAGS_MODE = 
else
CXXFLAGS_MODE  = -O3 #-DGPU
CUDAFLAGS_MODE = -O3 -m64 -gencode arch=compute_20,code=sm_20 
endif
#
export CXXFLAGS  += $(CXXFLAGS_MODE) -Wno-deprecated -std=c++0x -frounding-math -DCGAL_EIGEN3_ENABLED -DDEBUG_UCSF #-DDEBUG_TRACE
export CUDAFLAGS += $(CUDAFLAGS_MODE)
export UFL = ffc

# Warning: -Wno-deprecate might cause difficult linking issue difficult to solve

####################
## TIER LIBRARIES ##
####################
CGAL_DIR  = $(PATH_SOFT)/CGAL/
NIFTI_DIR = $(PATH_SOFT)/nifti
VTK_DIR   = $(PATH_SOFT)/VTK
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
export CUDA_LIB = /usr/local/cuda-5.0/lib64/

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
EXEC = $(MESH_RENDERING)/build_inrimage  $(FEM_MODELS_DIR)/Poisson

###############
## EXECUTION ##
###############
all: $(EXEC)

$(EXEC):
	( cd $(UTILS_DIR)/pugi/ && $(MAKE) )
	( cd $(FEM_MODELS_DIR) && $(MAKE) )
	( cd $(MESH_RENDERING_DIR) && $(MAKE) )
	@echo""
	@echo "export LD_LIBRARY_PATH=$(CUDA_LIB):$(VTK)/lib/vtk-5.10:$(CGAL)/lib:$(LD_LIBRARY_PATH)"
	@echo""


models:
	( cd $(FEM_MODELS_DIR) && $(MAKE) models )


clean:
	( cd $(UTILS_DIR)/pugi/ && $(MAKE) $@ )
	( cd $(FEM_MODELS_DIR) && $(MAKE) $@ )
	( cd $(MESH_RENDERING_DIR) && $(MAKE) $@ )

distclean: clean
	find . -name *~      -exec rm {} \;
	find . -name SLS_model.h -exec rm {} \;
	find . -name SLD_model.h -exec rm {} \;
	find . -name tCS_model.h -exec rm {} \;
#	find . -name *.xml   -exec rm {} \;
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