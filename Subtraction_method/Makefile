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
UFL_SRC    = Poisson.ufl
UFL_HEADER = Poisson.h
SRC        = main.cxx  Conductivity.cxx Source.cxx Subtraction.cxx PDE_solver_parameters.cxx
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = poisson
#
#
#
all: $(EXEC) 


poisson: $(UFL_HEADER) $(OBJ) 
	$(CC) $(CXXFLAGS) $(INC) $(LIB_DIR) $(LIBS) $^ -o $@  


%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<


$(UFL_HEADER): $(UFL_SRC)
	$(UFL) $(UFLFLAG) $?

clean:
	rm $(EXEC) Poisson.h $(OBJ) #  poisson000000.vtu poisson.pvd
