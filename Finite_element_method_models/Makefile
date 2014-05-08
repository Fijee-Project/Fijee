UFLFLAG = -l dolfin
#
# Libraries and includes
#
INC     = -I$(FIJEE)/ -I$(FIJEE)/Utils/Minimizers/ -I$(FENICS)/include/ -I$(EIGEN3)/include/eigen3/ 
LIB_DIR = -L$(FIJEE)/Utils/pugi/ -L$(FENICS)/lib -L/usr/lib64/mpich/lib/ 
LIBS    = -ldolfin  -lpugixml -lboost_mpi #-lboost_math_tr1
#########
#LIBS   += -L$(VTK)/lib/$(VTK_VERSION)/ $(VTK_LIBS)

#
# Sources
#
UFL_SRC    = SLS_model.ufl SLD_model.ufl tCS_model.ufl tCS_field_model.ufl
UFL_HEADER = $(UFL_SRC:.ufl=.h)
SRC        = main.cxx  Conductivity.cxx Source.cxx Electrodes_injection.cxx Intensity.cxx Physics.cxx SL_subtraction.cxx tCS_tDCS.cxx tCS_tACS.cxx tCS_tDCS_local_conductivity.cxx PDE_solver_parameters.cxx Spheres_electric_monopole.cxx Electrodes_setup.cxx Electrodes_surface.cxx #Spheres_electric_dipole.cxx
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = poisson
#
#
#
all: $(EXEC) 


poisson: $(OBJ) 
	$(CC) $(CXXFLAGS) $(INC) $(LIB_DIR) $(LIBS) $^ -o $@  


%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<


models: $(UFL_HEADER)
#	$(UFL) $(UFLFLAG) $^

%.h: %.ufl
	$(UFL) $(UFLFLAG) $<

clean:
	rm $(EXEC)  $(OBJ) #  poisson000000.vtu poisson.pvd
