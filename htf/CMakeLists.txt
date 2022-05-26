# Maintainer: Andrew White

if(NOT BUILD_MD)
    message(FATAL_ERROR "MPCD package cannot be built without MD.")
endif(NOT BUILD_MD)

set(COMPONENT_NAME htf)

set(_${COMPONENT_NAME}_sources
    module.cc
    CommStruct.cc
    TensorflowCompute.cc
    TFArrayComm.cc
    )
  
if(ENABLE_CUDA)
set(_${COMPONENT_NAME}_cu_sources
    TensorflowCompute.cu
    TFArrayComm.cu
    )
endif(ENABLE_CUDA)

  # check if we have python dependencies for compiling
  set(TF_REQUIRE_COMPILER "")
  if(NOT IGNORE_HTF_COMPILER)
    set(TF_REQUIRE_COMPILER ${CMAKE_CXX_COMPILER_VERSION})
  endif()
set(TF_REQUIRE_VERSION "2.3")
execute_process(COMMAND ${PYTHON_EXECUTABLE} -W ignore ${CMAKE_CURRENT_SOURCE_DIR}/check_tf_version.py ${TF_REQUIRE_VERSION} ${TF_REQUIRE_COMPILER}
    RESULT_VARIABLE ret)
if(ret EQUAL "1")
    message(FATAL_ERROR "-- Could not find suitable tensorflow or compiler to build ${COMPONENT_NAME} (see messages above)")
else()
    message("-- Found suitable tensorflow version for minimum ${TF_REQUIRE_VERSION}")
endif()




# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_${COMPONENT_NAME}_sources} ${_${COMPONENT_NAME}_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

#if(ENABLE_CUDA)
#CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${COMPONENT_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
#endif(ENABLE_CUDA)

add_library (_${COMPONENT_NAME}_static STATIC ${_${COMPONENT_NAME}_sources} ${${_COMPONENT_NAME}_cu_sources})
pybind11_add_module (_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${${_COMPONENT_NAME}_cu_sources} NO_EXTRAS)
# add visilibty flag, which is missing from hoomd?
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

if(APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
if(NOT HOOMD_MD_LIB)
  set(HOOMD_MD_LIB _md)
endif()
target_link_libraries(_${COMPONENT_NAME} PRIVATE HOOMD::_hoomd)
target_link_libraries(_${COMPONENT_NAME} PRIVATE HOOMD::_md)

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if(ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_${COMPONENT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_${COMPONENT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

if(ENABLE_CUDA)
fix_cudart_rpath(_${COMPONENT_NAME})
endif(ENABLE_CUDA)

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

##############################
# The Tensorflow

add_subdirectory(hoomd2tf_op)
add_subdirectory(tf2hoomd_op)

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${COMPONENT_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    tensorflowcompute.py
    version.py
    layers.py
    simmodel.py
    utils.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

copy_files_to_build("${files}" "${COMPONENT_NAME}" "*.py")

#add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})

if(BUILD_TESTING)
    add_subdirectory(test-py)
endif()

