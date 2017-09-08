cmake_minimum_required (VERSION 2.6)
project (EXACT)

# The version number.
set (EXACT_VERSION_MAJOR 0)
set (EXACT_VERSION_MINOR 32)

#add_definitions( -DEXACT_VERSION="${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR}" )

SET (PLATFORM 64)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -fsanitize=address -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -DNAN_CHECKS")
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
    #if we're compiling the client, don't look for MYSQL or TIFF libraries
    #to compile client add -DCOMPILE_CLIENT:STRING="YES" to the command line

ELSE (COMPILE_CLIENT STREQUAL "YES")
    find_package(MySQL)

    MESSAGE(STATUS "MYSQL_FOUND: ${MYSQL_FOUND}")
    IF (MYSQL_FOUND)
        add_definitions( -D_MYSQL_ )

        message(STATUS "including MYSQL_INCLUDE_DIR: ${MYSQL_INCLUDE_DIR}")
        include_directories(${MYSQL_INCLUDE_DIR})
    ENDIF (MYSQL_FOUND)

    find_package(tiff)

    message(STATUS "TIFF found? ${TIFF_FOUND}")
    message(STATUS "TIFF libraries: ${TIFF_LIBRARIES}")
    message(STATUS "TIFF include dir: ${TIFF_INCLUDE_DIR}")
    IF (TIFF_FOUND)
        add_definitions( -D_HAS_TIFF_ )
        include_directories(${TIFF_INCLUDE_DIR})
    ENDIF (TIFF_FOUND)
ENDIF (COMPILE_CLIENT STREQUAL "YES")

add_subdirectory(common)
add_subdirectory(image_tools)
add_subdirectory(time_series)
add_subdirectory(cnn)
add_subdirectory(cnn_tests)
add_subdirectory(cnn_examples)

FIND_PATH(TAO_INCLUDE_DIR ./examples/benchmarks.hxx
    ../tao #add addiitonal paths here if undvc_common is in a different directory
    )

IF (${TAO_INCLUDE_DIR} STREQUAL "TAO_INCLUDE_DIR-NOTFOUND") 
    SET (TAO_FOUND FALSE)
    MESSAGE(STATUS "TAO NOT FOUND!")
    SET(TAO_INCLUDE_DIR "")

ELSE (${TAO_INCLUDE_DIR} STREQUAL "TAO_INCLUDE_DIR-NOTFOUND") 
    SET (TAO_FOUND TRUE)

    MESSAGE(STATUS "TAO include directory: ${TAO_INCLUDE_DIR}")

    FIND_LIBRARY(TAO_MPI_LIBRARY NAMES mpi_algorithms PATHS ../tao/build/mpi PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_SYNCH_LIBRARY NAMES synchronous_algorithms PATHS ../tao/build/synchronous_algorithms PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_ASYNCH_LIBRARY NAMES asynchronous_algorithms PATHS ../tao/build/asynchronous_algorithms PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_DB_ASYNCH_LIBRARY NAMES db_asynchronous_algorithms PATHS ../tao/build/asynchronous_algorithms PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_NN_LIBRARY NAMES neural_networks PATHS ../tao/build/neural_networks PATH_SUFFIXES lib)
    FIND_LIBRARY(TAO_UTIL_LIBRARY NAMES tao_util PATHS ../tao/build/util  PATH_SUFFIXES lib)
    set(TAO_LIBRARY ${TAO_MPI_LIBRARY} ${TAO_SYNCH_LIBRARY} ${TAO_ASYNCH_LIBRARY} ${TAO_DB_ASYNCH_LIBRARY} ${TAO_NN_LIBRARY} ${TAO_UTIL_LIBRARY})
    MESSAGE(STATUS "TAO library: ${TAO_LIBRARY}")

ENDIF (${TAO_INCLUDE_DIR} STREQUAL "TAO_INCLUDE_DIR-NOTFOUND") 

include_directories(${TAO_INCLUDE_DIR})

IF (TAO_FOUND)
    add_subdirectory(rnn)
    #add_subdirectory(rnn_tests)
    add_subdirectory(rnn_examples)
ENDIF (TAO_FOUND)

add_subdirectory(opencl)

IF (COMPILE_CLIENT STREQUAL "YES")
    if (BOINC_APP_FOUND)
        message(STATUS "BOINC APP FOUND!")
        include_directories(${BOINC_INCLUDE_DIR})
        include_directories(${BOINC_INCLUDE_DIR}/api)
        include_directories(${BOINC_INCLUDE_DIR}/lib)

        add_subdirectory(client)
    ENDIF (BOINC_APP_FOUND)
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
