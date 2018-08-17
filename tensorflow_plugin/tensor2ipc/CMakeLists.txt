set(COMPONENT_NAME tensor2ipc_op)
set(_${COMPONENT_NAME}_sources
    tensor2ipc.cc
    )
set(_${COMPONENT_NAME}_cu_sources
    tensor2ipc.cu
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
find_package(CUDA 9.0 EXACT REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "TF_CFLAGS: ${TF_CFLAGS}")
message(STATUS "TF_LFLAGS: ${TF_LFLAGS}")
message(STATUS "HOOMD_CUDA_FLAGS: ${CUDA_ADDITIONAL_OPTIONS}")
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}/../../")


#Add TF specific options
SET(TF_CUDA_ADDITIONAL_OPTIONS ${CUDA_ADDITIONAL_OPTIONS} ${TF_CFLAGS} "-D GOOGLE_CUDA=1 --expt-relaxed-constexpr")
list(APPEND TF_CFLAGS -lcudart -D GOOGLE_CUDA=1)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${COMPONENT_NAME}_cu_sources} OPTIONS ${TF_CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

#add library target
add_library (_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_CUDA_GENERATED_FILES})
#set the TF compiler/linker flags
target_compile_options(_${COMPONENT_NAME} PRIVATE ${TF_CFLAGS})
target_link_libraries(_${COMPONENT_NAME} ${TF_LFLAGS})
target_link_libraries(_${COMPONENT_NAME} ${HOOMD_LIBRARIES})


# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
        )
