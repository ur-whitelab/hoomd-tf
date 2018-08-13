add_library (_ipc_array_comm SHARED IPCArrayComm.cc)

# link the library to its dependencies
target_link_libraries(_ipc_array_comm ${HOOMD_LIBRARIES})
set_target_properties(_ipc_array_comm PROPERTIES PREFIX "" OUTPUT_NAME "_ipc_array_comm")