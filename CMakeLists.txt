# Maintanier: HOOMD-TF Developers

project(htf)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_INSTALL_RPATH "$ORIGIN/..:$ORIGIN")

find_package(HOOMD 3.0 REQUIRED)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

set(PYTHON_SITE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install plugin to: " ${PYTHON_SITE_INSTALL_DIR})

# add subdirectories
add_subdirectory(${PROJECT_NAME})
