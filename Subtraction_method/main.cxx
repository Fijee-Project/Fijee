#include <vector>
#include <memory>
#include <dolfin.h>
//
// UCSF
//
#include "Physical_model.h"
#include "Subtraction.h"
//#include "Poisson.h"
//#include "Source.h"
//#include "Conductivity.h"
//#include "Boundaries.h"
//#include "Sub_Domaines.h"

using namespace dolfin;

int main()
{
  //
  // Parameters
  //
  //  parameters["num_threads"] = 4;
  // Allowed values are: [PETSc, STL, uBLAS, Epetra, MTL4].
  // Epetra in Trilinos
  // uBLAS needs UMFPACK
  parameters["linear_algebra_backend"] = "uBLAS";
  //  info(solver.parameters,true) ;
  //  info(parameters,true) ;
  //
  // Create mesh and function space
  //

  //
  // 
  std::unique_ptr< Solver::Physical_model> model( new Solver::Subtraction() );

  //
  //
  std::cout << "Loop over solvers" << std::endl;
  model->solver_loop();

//    //
//    //
//    //  Mesh mesh("/home/cobigo/devel/C++/UCSF/Bucket/mesh.xml");
//    std::cout << "Load the mesh" << std::endl;
//    Mesh mesh("../Mesh_rendering/mesh.xml");
//    info(mesh);
//    Poisson::FunctionSpace V(mesh);
//    
//    //
//    // Define boundary condition
//    Periphery perifery;
//  
//    //
//    // Define Subdomaines
//    std::cout << "Load Sub_domains" << std::endl;
//    //  MeshFunction< long unsigned int > domains(mesh, "/home/cobigo/devel/C++/UCSF/Bucket/mesh_subdomains.xml");
//    MeshFunction< long unsigned int > domains(mesh, "../Mesh_rendering/mesh_subdomains.xml");
//    //  CellFunction< uint > * domains = std::dynamic_cast< CellFunction< uint >* >(domains_file);
//  
//    //
//    // Initialize mesh function for boundary domains
//    // we tag the boundaries
//    FacetFunction< size_t > boundaries(mesh);
//    boundaries.set_all(0);
//    perifery.mark(boundaries, 1);
//  
//    //
//    // Define input data
//    // 
//    
//    //
//    // Conductivity
//    //
//    
//    std::cout << "Load Conductivity" << std::endl;
//    //
//    // Isotrope conductivity
//    Solver::Sigma_isotrope a_inf(0.33);
//  
//    //
//    // Anisotrope conductivity
//    Solver::Tensor_conductivity sigma(mesh);
//    //Solver::Sigma_isotrope sigma(0.43);
//  
//    //
//    // Source in an infinite space
//    Solver::Phi Phi_0;
//  
//    //
//    // Define Dirichlet boundary conditions 
//    DirichletBC bc(V, Phi_0, perifery);
//  
//    //
//    // Poisson equation
//    //
//  
//    //
//    // Define variational forms
//    Poisson::BilinearForm a(V, V);
//    Poisson::LinearForm L(V);
//  
//    //
//    // Anisotropy
//    // Bilinear
//    a.a_sigma = sigma;
//    //
//    a.dx = domains;
//    // Linear
//    L.a_inf   = a_inf;
//    L.a_sigma = sigma;
//    L.Phi_0   = Phi_0;
//    //
//    L.dx    = domains;
//    L.ds    = boundaries;
//  
//    //
//    // Compute solution
//    Function u(V);
//    // Solver
//    LinearVariationalProblem problem(a, L, u);
//    LinearVariationalSolver  solver(problem);
//    // krylov
//  
//  //    krylov_solver            |    type  value          range  access  change
//  //    ------------------------------------------------------------------------
//  //    absolute_tolerance       |  double  1e-15             []       0       0
//  //    divergence_limit         |  double  10000             []       0       0
//  //    error_on_nonconvergence  |    bool   true  {true, false}       0       0
//  //    maximum_iterations       |     int  10000             []       0       0
//  //    monitor_convergence      |    bool  false  {true, false}       0       0
//  //    nonzero_initial_guess    |    bool  false  {true, false}       0       0
//  //    relative_tolerance       |  double  1e-06             []       0       0
//  //    report                   |    bool   true  {true, false}       0       0
//  //    use_petsc_cusp_hack      |    bool  false  {true, false}       0       0
//  //
//    solver.parameters["linear_solver"]  = "cg";
//    solver.parameters("krylov_solver")["maximum_iterations"] = 20000;
//  //  solver.parameters["linear_solver"]  = "bicgstab";
//  //  solver.parameters["linear_solver"]  = "cg";
//    solver.parameters["preconditioner"] = "ilu";
//    // Cholesky
//  //  solver.parameters["linear_solver"]  = "umfpack";
//    solver.solve();
//  // //  // solve(a == L, u, bc);
//  //  solve(a == L, u);
//  // //  solve(a == L, u,
//  // //	solver_parameters = ("linear_solver":"cg",
//  // //			     "preconditioner":"ilu"));
//    
//    //
//    // View the solution
//    //
//    // Save solution in VTK format
//    File file("poisson.pvd");
//  
//  //  //
//  //  // Theoric potential in an infinit medium
//  //  Function Phi0_th(V);
//  //  Phi0_th.interpolate(Phi_0);
//  //  plot(Phi0_th, "Theoritical Potential");
//  //  file << Phi0_th;
//  //
//  //  Function Potential_total(V);
//  //  *Potential_total.vector()  = *u.vector();
//  //  *Potential_total.vector() += *Phi0_th.vector();
//  //  //  *diff.vector() /= *phi0_th.vector();
//  //  file << Potential_total;
//    file << u;
//    file << domains;
//  ////  // Plot solution
//  ////  plot(u, "Phi_s");
//  ////  plot(Potential_total, "Total Potential");
//  ////  plot(boundaries);
//  //  plot(domains);
//  //  //  plot(mesh);
//  //  interactive();

  //
  //
  return EXIT_SUCCESS;
}
