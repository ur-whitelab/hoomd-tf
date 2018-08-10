add_executable (ipccomm_test ipccomm_test.cc)

# link the library to its dependencies
target_link_libraries(ipccomm_test ${HOOMD_LIBRARIES})