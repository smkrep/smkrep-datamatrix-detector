cmake_minimum_required(VERSION 3.20)

project(smkrep-datamatrix-detector)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin.dbg)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin.rel)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin.relwithdbg)

find_package(OpenCV CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(detector detector.cpp)
target_link_libraries(detector PRIVATE opencv_core opencv_imgproc opencv_highgui)


######installation directly to build folder for proper debugging in VisualStudio####

####################################################################################
install(TARGETS detector DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG} CONFIGURATIONS Debug)
install(FILES ./img_dataset/img_dataset.rar DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/img_dataset CONFIGURATIONS Debug)
install(TARGETS detector DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE} CONFIGURATIONS Release)
install(FILES ./img_dataset/img_dataset.rar DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/img_dataset CONFIGURATIONS Release)
####################################################################################



######normal installation in a folder specified by CMAKE_INSTALL_PREFIX#############

####################################################################################
install(TARGETS detector DESTINATION . CONFIGURATIONS Debug Release)
install(FILES ./img_dataset/img_dataset.rar DESTINATION ./img_dataset CONFIGURATIONS Debug Release)
####################################################################################

