CXXFLAGS += -Wall -DTRACE=0 -DTIME_PROFILER 
#
# Libraries and includes
#
# Fijee
INC     = -I$(FIJEE)/ -I$(FIJEE)/Utils/ -I$(EIGEN3)/include/eigen3/
# Linbs
LIB_DIR = -L$(FIJEE)/Utils/pugi/ #-L$(FIJEE)/Utils/Minimizers/ -L$(FENICS)/lib -L/usr/lib64/mpich/lib/ 
LIBS    = -lpugixml #-ldolfin  -lfijee_minimizers -lboost_mpi #-lboost_math_tr1
#########
#LIBS   += -L$(VTK)/lib/$(VTK_VERSION)/ $(VTK_LIBS)

#
# Sources
#
SRC        = main.cxx Inverse_parameters.cxx #Inverse_solver.cxx #Empirical_bayesian_model.cxx 
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = inverse
#
#
#
all: $(EXEC) 


$(EXEC): $(OBJ) 
	$(CC) $(CXXFLAGS) $(INC) $(LIB_DIR) $(LIBS) $^ -o $@  


%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<


clean:
	rm $(EXEC)  $(OBJ) 
