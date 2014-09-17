set(INRIMAGE_SOURCES main.cxx VTK_implicite_domain.cxx  Spheres_implicite_domain.cxx  Head_labeled_domain.cxx Spheres_labeled_domain.cxx Build_mesh.cxx Access_parameters.cxx Spheres_conductivity_tensor.cxx Head_conductivity_tensor.cxx Point_vector.cxx Dipole.cxx Electrode.cxx Electrode_shape_sphere.cxx Electrode_shape_cylinder.cxx Cell_conductivity.cxx Point.cxx Distance.cxx Build_dipoles_list.cxx Build_electrodes_list.cxx Build_dipoles_list_knn.cxx)

add_executable(inrimage ${INRIMAGE_SOURCES})

target_link_libraries(inrimage ${LIBS} pugi CGAL_Qt4 CGAL_ImageIO) # CGAL cmake truncates these libraries for some reason
