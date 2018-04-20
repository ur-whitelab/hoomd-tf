set(COMPONENT_NAME tensorflow_op)
set(_${COMPONENT_NAME}_sources
    hoomd_op.cc
    )
set(_${COMPONENT_NAME}_cu_sources
    hoomd_op.cu
    )


#get TF specific compiler flags
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TF_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TF_LFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

#separate flags
SEPARATE_ARGUMENTS( TF_CFLAGS )
SEPARATE_ARGUMENTS( TF_LFLAGS )

if (ENABLE_CUDA)
#Add TF specific options
SET(TF_CUDA_ADDITIONAL_OPTIONS ${CUDA_ADDITIONAL_OPTIONS} ${TF_CFLAGS} "-DGOOGLE_CUDA")
list(APPEND TF_CFLAGS "-lcudart")
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