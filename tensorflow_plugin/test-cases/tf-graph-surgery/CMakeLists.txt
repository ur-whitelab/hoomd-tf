add_library (_ipc_tester SHARED ClassMethod.cc)

# link the library to its dependencies
target_link_libraries(_ipc_tester ${HOOMD_LIBRARIES})
set_target_properties(_ipc_tester PROPERTIES PREFIX "" OUTPUT_NAME "_ipc_tester")