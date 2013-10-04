#
# TRACE:
# 1, 2, 3, ... printout
# 100 R statistical software studies
# 200 VTK visualization
#
CXXFLAGS +=  -Wall -DTRACE=100 
INC       = -I$(FIJEE)/ -I$(NIFTI)/include/nifti/ -I$(VTK)/include/vtk-5.10/ -I/$(CGAL)/include/ -I$(EIGEN3)/include/eigen3/ #-I/usr/include/taucs/ #
LIBS      = -L$(FIJEE)/Utils/ -lpugixml -L/$(CGAL)/lib/ -lCGAL -lCGAL_ImageIO -lgmp -lboost_thread-mt -lmpfr #-lgsl -lgslcblas -lm
# Taucs
#LIBS += -L/usr/lib64/taucs/ -ltaucs -lblas -llapack -lmetis
#########
LIBS += -L$(VTK)/lib/vtk-5.10/ -lvtkCommon -lvtkFiltering -lvtkGenericFiltering -lvtkGraphics \
	-lvtkIO -lvtkRendering -lvtksys -lvtkVolumeRendering -lvtkzlib -lvtkfreetype  -lvtkftgl \
	-lvtkImaging  -lvtkhdf5  -lvtkhdf5_hl  -lvtkexpat  -lvtktiff  -lvtkjpeg  -lvtkpng  -lvtksqlite \
	-lvtkmetaio  -lLSDyna  -lvtkNetCDF   -lvtkDICOMParser  -lvtkverdict -lvtkNetCDF -lvtkHybrid
##########
# OpenMP #
#CXXFLAGS += -fopenmp 
##########
# CUDA   #
LIBS += -L$(CUDA_LIB) -lcudart
INC_CUDA = #-I$(NIFTI)/include/nifti/

######
######
SRC = main.cxx VTK_implicite_domain.cxx  Build_labeled_domain.cxx Build_mesh.cxx Access_parameters.cxx Conductivity_tensor.cxx Point_vector.cxx Dipole.cxx Cell_conductivity.cxx Point.cxx Distance.cxx #CUDA_Conductivity_matching.cxx
OBJ = $(SRC:.cxx=.o)
#
SRC_CUDA = CUDA_Conductivity_matching.cu
OBJ_CUDA = $(SRC_CUDA:.cu=.o)

EXEC = build_inrimage


all: $(EXEC) 

build_inrimage: $(OBJ) $(OBJ_CUDA)
	$(CXX) $(CXXFLAGS) $(INC) $(LIBS) $^ -o $@  

%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<

$(OBJ_CUDA): $(SRC_CUDA)
	$(CUDA) $(CUDAFLAGS) $(INC_CUDA) -o $@ -c $<



clean: 
	rm build_inrimage $(OBJ) $(OBJ_CUDA)
