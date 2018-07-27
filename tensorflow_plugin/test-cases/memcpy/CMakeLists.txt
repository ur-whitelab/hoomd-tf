add_executable (memcpy_test memcpy_test.cc)

# link the library to its dependencies
target_link_libraries(memcpy_test ${HOOMD_LIBRARIES})