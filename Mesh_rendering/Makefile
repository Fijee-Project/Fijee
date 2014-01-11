#
# TRACE:
# 1, 2, 3, ... printout
# 100 R statistical software studies
# 200 VTK visualization
#
CXXFLAGS +=  -Wall -DTRACE=0 
INC       = -I$(FIJEE)/ -I$(NIFTI)/include/nifti -I$(VTK)/include/vtk-5.10/ -I/$(CGAL)/include/ -I$(EIGEN3)/include/eigen3/ #-I/usr/include/taucs/ #
LIBS      = -L$(FIJEE)/Utils/pugi/ -lpugixml -L/$(CGAL)/lib/ -lCGAL -lCGAL_ImageIO -lgmp -lboost_thread-mt -lmpfr #-lgsl -lgslcblas -lm
# Taucs
#LIBS += -L/usr/lib64/taucs/ -ltaucs -lblas -llapack -lmetis
#########
LIBS += -L$(VTK)/lib/vtk-5.10/ -lvtkCommon -lvtkFiltering -lvtkGenericFiltering -lvtkGraphics \
	-lvtkIO -lvtkRendering -lvtksys -lvtkVolumeRendering -lvtkzlib -lvtkfreetype  -lvtkftgl \
	-lvtkImaging  -lvtkhdf5  -lvtkhdf5_hl  -lvtkexpat  -lvtktiff  -lvtkjpeg  -lvtkpng  -lvtksqlite \
	-lvtkmetaio  -lLSDyna  -lvtkNetCDF   -lvtkDICOMParser  -lvtkverdict -lvtkNetCDF -lvtkHybrid \
	-lvtkNetCDF_cxx -lvtkexoIIc
##########
# OpenMP #
#CXXFLAGS += -fopenmp 
##########
# CUDA   #
LIBS += -L$(CUDA_LIB) -lcudart
INC_CUDA = #-I$(NIFTI)/include/nifti/

######
######
SRC = main.cxx VTK_implicite_domain.cxx  Spheres_implicite_domain.cxx  Head_labeled_domain.cxx Spheres_labeled_domain.cxx Build_mesh.cxx Access_parameters.cxx Spheres_conductivity_tensor.cxx Head_conductivity_tensor.cxx Point_vector.cxx Dipole.cxx Electrode.cxx Cell_conductivity.cxx Point.cxx Distance.cxx Build_dipoles_list.cxx Build_electrodes_list.cxx Build_dipoles_list_knn.cxx ##CUDA_Conductivity_matching.cxx
OBJ = $(SRC:.cxx=.o)
#
SRC_CUDA = CUDA_Conductivity_matching.cu
OBJ_CUDA = $(SRC_CUDA:.cu=.o)

EXEC = build_inrimage


all: $(EXEC) 

build_inrimage: $(OBJ) $(OBJ_CUDA)
	$(CC) $(CXXFLAGS) $(INC) $(LIBS) $^ -o $@

%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<

$(OBJ_CUDA): $(SRC_CUDA)
	$(CUDA) $(CUDAFLAGS) $(INC) $(INC_CUDA) -o $@ -c $<



clean: 
	rm build_inrimage $(OBJ) $(OBJ_CUDA)
