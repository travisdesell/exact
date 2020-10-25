# - Find OpenCV 
# Find the native OpenCV includes and libraries
#
#  OpenCV_INCLUDE_DIR - where to find mysql.h, etc.
#  OpenCV_LIBRARIES   - List of libraries when using MySQL.
#  OpenCV_FOUND       - True if MySQL found.

IF (OpenCV_INCLUDE_DIR)
    # Already in cache, be silent
    SET(OpenCV_FIND_QUIETLY TRUE)
ENDIF (OpenCV_INCLUDE_DIR)

FIND_PATH(OpenCV_INCLUDE_DIR opencv.hpp highgui.hpp core.hpp
    /usr/local/Cellar/opencv/2.4.10.1/include/opencv2
    /usr/local/Cellar/opencv/2.4.10.1/include/opencv2/core
    /usr/local/Cellar/opencv/2.4.10.1/include/opencv2/highgui
    /usr/local/Cellar/opencv/2.4.10.1/include/opencv2/imgproc
    /usr/local/Cellar/opencv/2.4.9/include/opencv2
    /usr/local/Cellar/opencv/2.4.9/include/opencv2/core
    /usr/local/Cellar/opencv/2.4.9/include/opencv2/highgui
    /usr/local/Cellar/opencv/2.4.9/include/opencv2/imgproc
    /opt/local/include
    /opt/local/include/opencv2
    /usr/local/include/opencv2/core
    /usr/local/include/opencv2/highgui
    /usr/local/include/opencv2/imgproc
    /usr/include/opencv2
    ~/opencv/include/opencv2
)

MESSAGE(STATUS "OpenCV include directory: ${OpenCV_INCLUDE_DIR}")

FIND_LIBRARY(OpenCV_CORE_LIBRARY
    NAMES opencv_core
    PATHS /opt/local/lib/ ~/opencv/build/lib/ /usr/local/include/opencv2/core/
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "OpenCV core library: ${OpenCV_CORE_LIBRARY}")

FIND_LIBRARY(OpenCV_HIGH_GUI_LIBRARY
    NAMES opencv_highgui
    PATHS /opt/local/lib ~/opencv/build/lib/ /usr/local/include/opencv2/highgui
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "OpenCV high gui library: ${OpenCV_HIGH_GUI_LIBRARY}")

FIND_LIBRARY(OpenCV_IMGPROC_LIBRARY
    NAMES opencv_imgproc
    PATHS /opt/local/lib ~/opencv/build/lib/ /usr/local/include/opencv2/imgproc/
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "OpenCV imageproc library: ${OpenCV_IMGPROC_LIBRARY}")


IF (OpenCV_INCLUDE_DIR AND OpenCV_CORE_LIBRARY AND OpenCV_HIGH_GUI_LIBRARY AND OpenCV_IMGPROC_LIBRARY)
    add_definitions( -D_OpenCV_ )
    SET(OpenCV_FOUND TRUE)
    SET( OpenCV_LIBRARIES ${OpenCV_CORE_LIBRARY} ${OpenCV_HIGH_GUI_LIBRARY} ${OpenCV_IMGPROC_LIBRARY})
ELSE (OpenCV_INCLUDE_DIR AND OpenCV_CORE_LIBRARY AND OpenCV_HIGH_GUI_LIBRARY AND OpenCV_IMGPROC_LIBRARY)
    SET(OpenCV_FOUND FALSE)
    SET( OpenCV_LIBRARIES )
ENDIF (OpenCV_INCLUDE_DIR AND OpenCV_CORE_LIBRARY AND OpenCV_HIGH_GUI_LIBRARY AND OpenCV_IMGPROC_LIBRARY)

IF (OpenCV_FOUND)
    IF (NOT OpenCV_FIND_QUIETLY)
        MESSAGE(STATUS "Found OpenCV: ${OpenCV_LIBRARIES}")
        MESSAGE(STATUS " -- OpenCV include directory: ${OpenCV_INCLUDE_DIR}")
    ENDIF (NOT OpenCV_FIND_QUIETLY)
ELSE (OpenCV_FOUND)
    IF (OpenCV_FIND_REQUIRED)
        MESSAGE(STATUS "Looked for OpenCV libraries named ${OpenCV_NAMES}.")
        MESSAGE(FATAL_ERROR "Could NOT find OpenCV library")
    ENDIF (OpenCV_FIND_REQUIRED)
ENDIF (OpenCV_FOUND)

MARK_AS_ADVANCED(
    OpenCV_CORE_LIBRARY
    OpenCV_HIGH_GUI_LIBRARY
    OpenCV_IMGPROC_LIBRARY
    OpenCV_INCLUDE_DIR
    )
