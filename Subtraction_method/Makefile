UFLFLAG = -l dolfin
#
# Libraries and includes
#
INC     = -I$(FENICS)/include/
LIB_DIR = -L/usr/lib64/mpich2/lib/ -L$(FENICS)/lib 
LIBS    = -lboost_mpi -ldolfin
#
# Sources
#
UFL_SRC    = Poisson.ufl
UFL_HEADER = Poisson.h
SRC        = 
OBJ        = 
#
#
#
EXEC = poisson
#
#
#
all: $(OBJ) $(UFL_HEADER)
	$(CXX) $(CXXFLAGS) $(INC) main.cxx $(LIB_DIR) $(LIBS) -o $(EXEC)

$(UFL_HEADER): $(UFL_SRC)
	$(UFL) $(UFLFLAG) $?

clean:
	rm $(EXEC) Poisson.h # *.o poisson000000.vtu poisson.pvd
