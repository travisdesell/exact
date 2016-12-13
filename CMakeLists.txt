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

#message("project source dir = ${PROJECT_SOURCE_DIR}")

#This will look for the undvc_common library, which should be in the
#same directory as this project. It should be compiled in the ../undvc_common/build
#directory
FIND_PATH(UNDVC_COMMON_INCLUDE_DIR file_io.hxx
    ../undvc_common         #add addiitonal paths here if undvc_common is in a different directory
    )
MESSAGE(STATUS "UNDVC_COMMON include directory: ${UNDVC_COMMON_INCLUDE_DIR}")

FIND_LIBRARY(UNDVC_COMMON_LIBRARY
    NAMES undvc_common
    PATHS ../undvc_common/build         #add additional paths here if undvc_common is in a different directory
                                        #or if it has been build in a different directory
    PATH_SUFFIXES lib
    )
MESSAGE(STATUS "UNDVC_COMMON library: ${UNDVC_COMMON_LIBRARY}")

include_directories(${UNDVC_COMMON_INCLUDE_DIR})

#UNDVC_COMMON has the FindBOINC.cmake and FindMYSQL.cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${UNDVC_COMMON_INCLUDE_DIR}/cmake/Modules/")

find_package(Boost REQUIRED COMPONENTS system filesystem)
MESSAGE(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
MESSAGE(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
MESSAGE(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")


find_package(BOINC)
#find_package(MySQL REQUIRED)
#find_package(OpenSSL REQUIRED)

IF (BOINC_APP_FOUND)
    message(STATUS "BOINC APP FOUND!")
    include_directories(${BOINC_INCLUDE_DIR})
    include_directories(${BOINC_INCLUDE_DIR}/api)
    include_directories(${BOINC_INCLUDE_DIR}/lib)

    add_definitions( -D_BOINC_ )
ENDIF (BOINC_APP_FOUND)


include_directories(
	${BOINC_INCLUDE_DIR}
    ${BOINC_INCLUDE_DIR}/api
    ${BOINC_INCLUDE_DIR}/db
    ${BOINC_INCLUDE_DIR}/lib
    ${BOINC_INCLUDE_DIR}/sched
    ${BOINC_INCLUDE_DIR}/tools/
	${MYSQL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

include_directories(${Boost_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(image_tools)
add_subdirectory(strategy)
add_subdirectory(client)

#if (BOINC_SERVER_FOUND)
#    add_subdirectory(server)
#ENDIF(BOINC_SERVER_FOUND)


add_subdirectory(multithreaded)
add_subdirectory(mpi)
