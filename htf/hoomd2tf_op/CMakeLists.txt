# Maintainer: Andrew White

set(SUB_COMPONENT_NAME hoomd2tf_op)
set(_${SUB_COMPONENT_NAME}_sources
    hoomd2tf.cc
    )
set(_${SUB_COMPONENT_NAME}_cu_sources
    hoomd2tf.cu
    )


#get TF specific compiler flags
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TF_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TF_LFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

#separate flags
SEPARATE_ARGUMENTS( TF_CFLAGS )
SEPARATE_ARGUMENTS( TF_LFLAGS )

if (ENABLE_CUDA)
#These have to be added (?)
# use cuda
find_package(CUDA 9.0)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}/../../")


#Add TF specific options
SET(TF_CUDA_ADDITIONAL_OPTIONS ${CUDA_ADDITIONAL_OPTIONS} ${TF_CFLAGS} "-D GOOGLE_CUDA=1 -DNDEBUG--expt-relaxed-constexpr")
list(APPEND TF_CFLAGS -lcudart -D GOOGLE_CUDA=1 -DNDEBUG)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "TF_CFLAGS: ${TF_CFLAGS}")
message(STATUS "TF_LFLAGS: ${TF_LFLAGS}")
message(STATUS "HOOMD_CUDA_FLAGS: ${CUDA_ADDITIONAL_OPTIONS}")
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${SUB_COMPONENT_NAME}_cu_sources} OPTIONS ${TF_CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

#add library target
add_library (_${SUB_COMPONENT_NAME} SHARED ${_${SUB_COMPONENT_NAME}_sources} ${_CUDA_GENERATED_FILES})
#set the TF compiler/linker flags
message(STATUS "HOOMD_TF_CUDA_C++_FLAGS: ${TF_CFLAGS}")
target_compile_options(_${SUB_COMPONENT_NAME} PRIVATE ${TF_CFLAGS})
target_link_libraries(_${SUB_COMPONENT_NAME} ${TF_LFLAGS})
target_link_libraries(_${SUB_COMPONENT_NAME} ${HOOMD_LIBRARIES})


# install the library
install(TARGETS _${SUB_COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}/${SUB_COMPONENT_NAME}
        )
