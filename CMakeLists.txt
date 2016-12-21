cmake_minimum_required (VERSION 2.6)
project (EXACT)

# The version number.
set (EXACT_VERSION_MAJOR 0)
set (EXACT_VERSION_MINOR 01)

SET (PLATFORM 64)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funrol-loops -DNDEBUG")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

message(STATUS "project source dir is ${PROJECT_SOURCE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

message (STATUS "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}")

find_package(BOINC)
MESSAGE(STATUS "BOINC_APP_FOUND: ${BOINC_APP_FOUND}")
MESSAGE(STATUS "BOINC_SERVER_FOUND: ${BOINC_SERVER_FOUND}")

IF (BOINC_APP_FOUND)
    message(STATUS "BOINC APP FOUND!")
    include_directories(${BOINC_INCLUDE_DIR})
    include_directories(${BOINC_INCLUDE_DIR}/api)
    include_directories(${BOINC_INCLUDE_DIR}/lib)
ENDIF (BOINC_APP_FOUND)

find_package(MySQL)

MESSAGE(STATUS "MYSQL_FOUND: ${MYSQL_FOUND}")
if (MYSQL_FOUND)
    add_definitions( -D_MYSQL_ )
ENDIF (MYSQL_FOUND)

include_directories(${PROJECT_SOURCE_DIR})

include_directories(
	${BOINC_INCLUDE_DIR}
    ${BOINC_INCLUDE_DIR}/api
    ${BOINC_INCLUDE_DIR}/db
    ${BOINC_INCLUDE_DIR}/lib
    ${BOINC_INCLUDE_DIR}/sched
    ${BOINC_INCLUDE_DIR}/tools/
	${MYSQL_INCLUDE_DIR}
)


add_subdirectory(common)
add_subdirectory(image_tools)
add_subdirectory(strategy)

if (BOINC_APP_FOUND)
    add_subdirectory(client)
ENDIF(BOINC_APP_FOUND)

if (BOINC_SERVER_FOUND)
    MESSAGE(STATUS "BOINC_SERVER_FOUND")

    MESSAGE(STATUS "OpenSSL required.")
    find_package(OpenSSL REQUIRED)

    add_subdirectory(server)
ENDIF(BOINC_SERVER_FOUND)

add_subdirectory(multithreaded)
add_subdirectory(mpi)
