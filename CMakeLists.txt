cmake_minimum_required (VERSION 2.6)
project (EXACT)

# The version number.
set (EXACT_VERSION_MAJOR 0)
set (EXACT_VERSION_MINOR 12)

add_definitions( -DEXACT_VERSION="${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR}" )

SET (PLATFORM 64)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funroll-loops -DNDEBUG")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

message(STATUS "project source dir is ${PROJECT_SOURCE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

message (STATUS "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}")

find_package(BOINC)
MESSAGE(STATUS "BOINC_APP_FOUND: ${BOINC_APP_FOUND}")
MESSAGE(STATUS "BOINC_SERVER_FOUND: ${BOINC_SERVER_FOUND}")

include_directories(${PROJECT_SOURCE_DIR})

SET(COMPILE_CLIENT "NO" CACHE STRING "Compile the BOINC client app or not")

MESSAGE(STATUS "COMPILE CLIENT SET TO: ${COMPILE_CLIENT}")

IF (COMPILE_CLIENT STREQUAL "YES")
    #if we're compiling the client, don't look for MYSQL libraries
    #to compile client add -DCOMPILE_CLIENT:STRING="YES" to the command line

ELSE (COMPILE_CLIENT STREQUAL "YES")
    find_package(MySQL)

    MESSAGE(STATUS "MYSQL_FOUND: ${MYSQL_FOUND}")
    IF (MYSQL_FOUND)
        add_definitions( -D_MYSQL_ )

        message(STATUS "including MYSQL_INCLUDE_DIR: ${MYSQL_INCLUDE_DIR}")
        include_directories(${MYSQL_INCLUDE_DIR})
    ENDIF (MYSQL_FOUND)
ENDIF (COMPILE_CLIENT STREQUAL "YES")


add_subdirectory(common)
add_subdirectory(image_tools)
add_subdirectory(strategy)
add_subdirectory(tests)

IF (COMPILE_CLIENT STREQUAL "YES")
    if (BOINC_APP_FOUND)
        message(STATUS "BOINC APP FOUND!")
        include_directories(${BOINC_INCLUDE_DIR})
        include_directories(${BOINC_INCLUDE_DIR}/api)
        include_directories(${BOINC_INCLUDE_DIR}/lib)

        add_subdirectory(client)
    ENDIF(BOINC_APP_FOUND)
ENDIF (COMPILE_CLIENT STREQUAL "YES")


if (BOINC_SERVER_FOUND)
    MESSAGE(STATUS "BOINC_SERVER_FOUND")

    MESSAGE(STATUS "OpenSSL required.")
    find_package(OpenSSL REQUIRED)

    include_directories(
        ${BOINC_INCLUDE_DIR}
        ${BOINC_INCLUDE_DIR}/api
        ${BOINC_INCLUDE_DIR}/db
        ${BOINC_INCLUDE_DIR}/lib
        ${BOINC_INCLUDE_DIR}/sched
        ${BOINC_INCLUDE_DIR}/tools/
        ${MYSQL_INCLUDE_DIR}
        )

    add_subdirectory(server)
ENDIF(BOINC_SERVER_FOUND)

IF (COMPILE_CLIENT STREQUAL "NO")
	add_subdirectory(multithreaded)
	add_subdirectory(mpi)
ENDIF (COMPILE_CLIENT STREQUAL "NO")
