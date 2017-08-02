add_library(exact_image_tools image_set large_image_set)

add_executable(convert_mnist_data convert_mnist_data)

add_executable(split_mnist_data split_mnist_data)

add_executable(convert_cifar10_data convert_cifar10_data)

#find_package(tiff)
#add_executable(large_image_set large_image_set)
#target_link_libraries(large_image_set ${TIFF_LIBRARIES})
#target_compile_definitions(large_image_set PUBLIC -DLARGE_IMAGES_TEST)
