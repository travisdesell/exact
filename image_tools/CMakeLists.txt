IF (TIFF_FOUND)
    add_library(exact_image_tools image_set large_image_set mosaic_image_set)

    add_executable(mosaic_image_set large_image_set mosaic_image_set)
    target_link_libraries(mosaic_image_set ${TIFF_LIBRARIES})
    target_compile_definitions(mosaic_image_set PUBLIC -DMOSAIC_IMAGES_TEST)

#ELSE (TIFF_FOUND)
    #add_library(exact_image_tools image_set large_image_set)
ENDIF (TIFF_FOUND)

add_executable(convert_mnist_data convert_mnist_data)

add_executable(split_mnist_data split_mnist_data)

add_executable(convert_cifar10_data convert_cifar10_data)

add_executable(large_image_set large_image_set)
target_link_libraries(large_image_set ${TIFF_LIBRARIES})
target_compile_definitions(large_image_set PUBLIC -DLARGE_IMAGES_TEST)

