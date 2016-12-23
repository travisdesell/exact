#include_directories(${BOINC_INCLUDE_DIR})
#include_directories(${BOINC_INCLUDE_DIR}/api)
#include_directories(${BOINC_INCLUDE_DIR}/lib)

#find_package(BOINC)

#add_definitions( -D_BOINC_ -DENABLE_CHECKPOINTING -DVERBOSE -DFALSE_ONLY -D__STDC_LIMIT_MACROS)

add_executable(exact_client_${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR} client)
target_link_libraries(exact_client_${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR} ${BOINC_APP_LIBRARIES} exact_common exact_strategy exact_image_tools)
