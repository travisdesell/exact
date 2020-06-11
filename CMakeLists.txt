cmake_minimum_required (VERSION 2.6)
project (EXACT)

# The version number.
set (EXACT_VERSION_MAJOR 0)
set (EXACT_VERSION_MINOR 33)

#add_definitions( -DEXACT_VERSION="${EXACT_VERSION_MAJOR}.${EXACT_VERSION_MINOR}" )

SET (PLATFORM 64)

#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -stdlib=libstdc++")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -fsanitize=address -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -g -Wall -O1 -funroll-loops -msse3 -fsanitize=address -fno-omit-frame-pointer -DNAN_CHECKS -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -D_GLIBCXX_DEBUG")
#SET (CMAKE_CXX_FLAGS                "-std=c++11 -Wall -O3 -funroll-loops -msse3 -DNAN_CHECKS")

# 1 This line for local
SET (CMAKE_CXX_FLAGS                "-std=c++17 -Wall -O3 -funroll-loops  -msse3 -fsanitize=address -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")
# 2 This line for cluster
SET (CMAKE_CXX_FLAGS                "-std=c++17 -Wall -O3 -funroll-loops  -msse3 -fno-omit-frame-pointer -D_GLIBCXX_DEBUG")

#SET (CMAKE_CXX_FLAGS                "-std=c++17 -Wall -O3 -funroll-loops -msse3")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funroll-loops -DNDEBUG")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

message(STATUS "project source dir is ${PROJECT_SOURCE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

message (STATUS "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}")

find_package(MPI)
#SET(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/3.1.2/include)
#SET(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/4.0.0/include)
SET(MPI_INCLUDE_PATH /usr/local/Cellar/open-mpi/4.0.1/include)
#SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/3.1.2/lib/libmpi.dylib)
#SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/4.0.0/lib/libmpi.dylib)
SET(MPI_LIBRARY /usr/local/Cellar/open-mpi/4.0.1/lib/libmpi.dylib)
MESSAGE(STATUS "MPI include directory: ${MPI_INCLUDE_PATH}")
MESSAGE(STATUS "MPI library: ${MPI_LIBRARY}")
MESSAGE(STATUS "MPI extra: ${MPI_EXTRA}")
include_directories(${MPI_INCLUDE_PATH})

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

    #set(TIFF_INCLUDE_DIR "/usr/include/x86_64-linux-gnu/")
    #set(TIFF_LIBRARIES /usr/lib/x86_64-linux-gnu/libtiffxx.so /usr/lib/x86_64-linux-gnu/libtiff.so)
    #set(TIFF_FOUND true)
    #set(TIFF_INCLUDE_DIR "/usr/include/")
    #set(TIFF_LIBRARIES /usr/lib64/libtiffxx.so /usr/lib64/libtiff.so)
    #set(TIFF_FOUND true)

    find_package(TIFF)

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

add_subdirectory(rnn)
add_subdirectory(rnn_tests)
add_subdirectory(rnn_examples)

#add_subdirectory(opencl)

add_subdirectory(cnn_tests)
add_subdirectory(cnn_examples)
add_subdirectory(multithreaded)
add_subdirectory(mpi)



IF (COMPILE_CLIENT STREQUAL "YES")
    if (BOINC_APP_FOUND)
        message(STATUS "BOINC APP FOUND!")
        include_directories(${BOINC_INCLUDE_DIR})
        include_directories(${BOINC_INCLUDE_DIR}/api)
        include_directories(${BOINC_INCLUDE_DIR}/lib)

        add_subdirectory(client)
    ENDIF (BOINC_APP_FOUND)
ENDIF (COMPILE_CLIENT STREQUAL "YES")


IF (BOINC_SERVER_FOUND)
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

    #    add_subdirectory(server)
ENDIF(BOINC_SERVER_FOUND)
