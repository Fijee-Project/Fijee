//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#ifndef FIJEE_ENVIRONMENT_H
#define FIJEE_ENVIRONMENT_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Fijee_environment.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <stdlib.h>     /* getenv */
#include <errno.h>      /* builtin errno */
#include <sys/stat.h>   /* mkdir */
#include <cstring>      /* strerror */
#include <exception>
//
// Get built-in type.  Creates member get_"name"() (e.g., get_visibility());
//
#define ucsf_get_macro(name,type) \
  inline type get_##name()const { \
    return this->name;		  \
  } 
//
// Get character string.  Creates member get_"name"() 
// (e.g., char *GetFilename());
//
#define ucsf_get_string_macro(name) \
  const char* get_##name() const {  \
    return this->name.c_str();	    \
  } 
//
//
//
typedef struct stat stat_file;
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Fijee_environment
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Fijee_environment
  {
  private:
    // Freesurfer path. 
    std::string subjects_dir_;
    std::string subject_;
    //! Finite element method directory path.
    std::string fem_path_;
    //! Finite element method output directory path.
    std::string fem_output_path_;
    //! Finite element method result directory path.
    std::string fem_result_path_;
    //! Finite element method measure directory path.
    std::string fem_measure_path_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Fijee_environment
     *
     */
    Fijee_environment()
      {
	//
	// Check on ENV variables
	// ToDo: replace getenv ...
	// getenv never throws exceptions
	subjects_dir_ = getenv ("SUBJECTS_DIR");
	subject_      = getenv ("SUBJECT");
	//
	if( subjects_dir_.empty() || subject_.empty() )
	  {
	    std::cerr << "FreeSurfer Env variables $SUBJECTS_DIR and $SUBJECT must be defined" 
		      << std::endl;
	    exit(1);
	  }
	//
	fem_path_ = std::string( subjects_dir_.c_str() ) + std::string("/") + std::string( subject_.c_str() ) + std::string("/");

	//
	// Create output path
	stat_file st;
	int       status = 0;
	mode_t    mode   = 0750;
	//
	fem_output_path_ = fem_path_ + std::string("/fem/output/");
	//
	if ( stat( fem_output_path_.c_str(), &st ) != 0 )
	  {
	    /* Directory does not exist */
	    if ( mkdir( fem_output_path_.c_str(), mode ) != 0 )
	      status = -1;
	  }
	else if (!S_ISDIR(st.st_mode))
	  {
	    errno = ENOTDIR;
	    status = -1;
	  }
	else
	  {
	    std::cerr << "Warning: directory " << fem_output_path_
		      << " already exist. Data will be removed." << std::endl;
	  }
	//
	if (status == -1 )
	  {
	    std::cerr << "failed to create " << fem_output_path_
		      << ": " << strerror(errno) << std::endl;
	    //
	    exit(1);
	  }

	//
	// Create result path
	status = 0;
	//
	fem_result_path_ = fem_path_ + std::string("/fem/result/");
	//
	if ( stat( fem_result_path_.c_str(), &st ) != 0 )
	  {
	    /* Directory does not exist */
	    if ( mkdir( fem_result_path_.c_str(), mode ) != 0 )
	      status = -1;
	  }
	else if (!S_ISDIR(st.st_mode))
	  {
	    errno = ENOTDIR;
	    status = -1;
	  }
	else
	  {
	    std::cerr << "Warning: directory " << fem_result_path_
		      << " already exist. Data will be removed." << std::endl;
	  }
	//
	if (status == -1 )
	  {
	    std::cerr << "failed to create " << fem_result_path_
		      << ": " << strerror(errno) << std::endl;
	    //
	    exit(1);
	  }

	//
	// append the path line
	fem_path_ += std::string("/");

	//
	// Create measure path
	status = 0;
	//
	fem_measure_path_ = fem_path_ + std::string("/fem/measure/");
	//
	if ( stat( fem_measure_path_.c_str(), &st ) != 0 )
	  {
	    /* Directory does not exist */
	    if ( mkdir( fem_measure_path_.c_str(), mode ) != 0 )
	      status = -1;
	  }
	else if (!S_ISDIR(st.st_mode))
	  {
	    errno = ENOTDIR;
	    status = -1;
	  }
	else
	  {
	    std::cerr << "Warning: directory " << fem_measure_path_
		      << " already exist. " << std::endl;
	  }
	//
	if (status == -1 )
	  {
	    std::cerr << "failed to create " << fem_measure_path_
		      << ": " << strerror(errno) << std::endl;
	    //
	    exit(1);
	  }

	//
	// append the path line
	fem_path_ += std::string("/");
      };
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Fijee_environment( const Fijee_environment& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Fijee_environment
     */
    virtual ~Fijee_environment(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Fijee_environment
     *
     */
    Fijee_environment& operator = ( const Fijee_environment& ){return *this;};

  public:
    /*!
     *  \brief Get subject_
     *
     *  This method return the subject name
     *
     */
    ucsf_get_macro(subject_, std::string);
   /*!
     *  \brief Get subjects_dir_
     *
     *  This method return the subjects directory
     *
     */
    ucsf_get_macro(subjects_dir_, std::string);
    /*!
     *  \brief Get fem_path_
     *
     *  This method return the finite element method directory path.
     *
     */
    ucsf_get_macro(fem_path_, std::string);
   /*!
     *  \brief Get fem_output_path_
     *
     *  This method return the finite element method output directory path.
     *
     */
    ucsf_get_macro(fem_output_path_, std::string);
   /*!
     *  \brief Get fem_result_path_
     *
     *  This method return the finite element method result directory path.
     *
     */
    ucsf_get_macro(fem_result_path_, std::string);
   /*!
     *  \brief Get fem_measure_path_
     *
     *  This method return the finite element method measure directory path.
     *
     */
    ucsf_get_macro(fem_measure_path_, std::string);

    
  };
  /*!
   *  \brief Dump values for Fijee_environment
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Fijee_environment& );
};
#endif
