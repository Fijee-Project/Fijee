UFLFLAG = -l dolfin
#
# Libraries and includes
#
INC     = -I$(FIJEE)/ -I$(FENICS)/include/ 
LIB_DIR = -L$(FIJEE)/Utils/pugi/ -L$(FENICS)/lib -L/usr/lib64/mpich2/lib/ #-L$(VTK)/lib/vtk-5.10/ 
LIBS    = -ldolfin -lboost_mpi -lpugixml 
#
# Sources
#
UFL_SRC    = SLS_model.ufl SLD_model.ufl
UFL_HEADER = $(UFL_SRC:.ufl=.h)
SRC        = main.cxx  Conductivity.cxx Source.cxx SL_subtraction.cxx PDE_solver_parameters.cxx
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
