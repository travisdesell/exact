# Install script for directory: C:/Users/micha/Documents/Masters Capstone/exact

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/micha/Documents/Masters Capstone/exact/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/common/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/image_tools/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/time_series/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/word_series/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/cnn/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/rnn/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/rnn_tests/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/rnn_examples/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/opencl/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/cnn_tests/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/cnn_examples/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/multithreaded/cmake_install.cmake")
  include("C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/mpi/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/micha/Documents/Masters Capstone/exact/out/build/x64-Debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
