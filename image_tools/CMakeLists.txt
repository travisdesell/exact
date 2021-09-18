IF (TIFF_FOUND)
    add_library(exact_image_tools lodepng.cpp image_set.cxx large_image_set.cxx mosaic_image_set.cxx)

    add_executable(mosaic_image_set lodepng.cpp large_image_set.cxx mosaic_image_set.cxx)
    target_link_libraries(mosaic_image_set ${TIFF_LIBRARIES})
    target_compile_definitions(mosaic_image_set PUBLIC -DMOSAIC_IMAGES_TEST)

#ELSE (TIFF_FOUND)
    #add_library(exact_image_tools image_set large_image_set)
ENDIF (TIFF_FOUND)

add_executable(convert_mnist_data convert_mnist_data.cxx)

add_executable(split_mnist_data split_mnist_data.cxx)

add_executable(convert_cifar10_data convert_cifar10_data.cxx)

add_executable(large_image_set lodepng.cpp large_image_set.cxx)
target_link_libraries(large_image_set ${TIFF_LIBRARIES})
target_compile_definitions(large_image_set PUBLIC -DLARGE_IMAGES_TEST)

