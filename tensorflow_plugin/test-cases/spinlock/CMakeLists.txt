add_library (_spinlock SHARED spinlock.cc)

# link the library to its dependencies
target_link_libraries(_spinlock ${HOOMD_LIBRARIES})
set_target_properties(_spinlock PROPERTIES PREFIX "" OUTPUT_NAME "_spinlock")