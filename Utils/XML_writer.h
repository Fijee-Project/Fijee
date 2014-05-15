#ifndef XML_WRITER_H
#define XML_WRITER_H
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file XML_writer.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <fstream>      // std::ifstream, std::ofstream
#include <sstream>
#include "Utils/pugi/pugixml.hpp"
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class XML_writer
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class XML_writer
  {
  protected:
    //! XML document
    pugi::xml_document document_;
    //! Root of the XML tree
    pugi::xml_node fijee_;
    //! XML file name
    std::string file_name_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class XML_writer
     *
     */
    XML_writer(){};
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class XML_writer
     *
     */
  XML_writer( std::string File_name ):
    file_name_( File_name )
      {
 	// Main node fijee
	fijee_ = document_.append_child("fijee");
	// 
	fijee_.append_attribute("xmlns:fijee") = "https://github.com/Fijee-Project/Fijee";
     };
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    XML_writer( const XML_writer& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class XML_writer
     */
    virtual ~XML_writer()
      {
	document_.save_file( file_name_.c_str() ); 
      };
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class XML_writer
     *
     */
    XML_writer& operator = ( const XML_writer& ){return *this;};
  };
};
#endif
