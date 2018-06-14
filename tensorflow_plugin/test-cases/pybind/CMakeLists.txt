add_library (_class_method SHARED ClassMethod.cc)

# link the library to its dependencies
target_link_libraries(_class_method ${HOOMD_LIBRARIES})
set_target_properties(_class_method PROPERTIES PREFIX "" OUTPUT_NAME "_class_method")