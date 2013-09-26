###############
## CONFIGURE ##
###############
export PREFIX =
DEBUG         = no
VERSION       = 1.0
DIST          = fijee

#################
## COMPILATION ##
#################
export CXX  = g++
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
CGAL_DIR  = /home/cobigo/devel/C++/CGAL
NIFTI_DIR = /home/cobigo/devel/C++/nifti
VTK_DIR   = /home/cobigo/devel/C++/VTK
#
ifeq ($(DEBUG),yes)
export CGAL   += $(CGAL_DIR)/install.debug
export NIFTI  += $(NIFTI_DIR)/install.debug
export VTK    += $(VTK_DIR)/install-5.10.1.debug
else
export CGAL   += $(CGAL_DIR)/install
export NIFTI  += $(NIFTI_DIR)/install
export VTK    += $(VTK_DIR)/install
endif
#
export EIGEN3   = /home/cobigo/devel/C++/Eigen3/install
export FENICS   = /home/cobigo/devel/CPP/FEniCS-project/install/
export CUDA_LIB = /usr/local/cuda-5.0/lib64

######################
## Main DIRECTORIES ##
######################
export FIJEE += $(CURDIR)

#####################
## SUB DIRECTORIES ##
#####################
UTILS_DIR              = Utils
MESH_RENDERING_DIR     = Mesh_rendering
SUBTRACTION_METHOD_DIR = Subtraction_method
EXEC = $(MESH_RENDERING)/build_inrimage  $(SUBTRACTION_METHOD_DIR)/Poisson

###############
## EXECUTION ##
###############
all: $(EXEC)

$(EXEC):
	( cd $(UTILS_DIR) && $(MAKE) )
	( cd $(SUBTRACTION_METHOD_DIR) && $(MAKE) )
	( cd $(MESH_RENDERING_DIR) && $(MAKE) )

clean:
	( cd $(UTILS_DIR) && $(MAKE) $@ )
	( cd $(SUBTRACTION_METHOD_DIR) && $(MAKE) $@ )
	( cd $(MESH_RENDERING_DIR) && $(MAKE) $@ )

distclean: clean
	find . -name *~      -exec rm {} \;
	find . -name *.xml   -exec rm {} \;
	find . -name *.mesh  -exec rm {} \;
	find . -name *.vtu   -exec rm {} \;
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
	mkdir $(DIST)/$(SUBTRACTION_METHOD_DIR)
	mkdir $(DIST)/$(MESH_RENDERING_DIR)
	cp $(SUBTRACTION_METHOD_DIR)/Makefile      $(DIST)/$(SUBTRACTION_METHOD_DIR)/
	cp $(MESH_RENDERING_DIR)/{Makefile,README} $(DIST)/$(MESH_RENDERING_DIR)/      
	cp $(SUBTRACTION_METHOD_DIR)/*.{h,cxx,ufl} $(DIST)/$(SUBTRACTION_METHOD_DIR)/
	cp $(MESH_RENDERING_DIR)/*.{h,cxx,cu}      $(DIST)/$(MESH_RENDERING_DIR)/      
	tar zcvf $(DIST)-$(VERSION).tar.gz $(DIST)
	rm -rf $(DIST)