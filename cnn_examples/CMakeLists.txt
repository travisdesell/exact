add_executable(evaluate_large_image_cnn evaluate_large_image_cnn)
target_link_libraries(evaluate_large_image_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(evaluate_cnn evaluate_cnn)
target_link_libraries(evaluate_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(one_layer one_layer)
target_link_libraries(one_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(two_layer two_layer)
target_link_libraries(two_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(lenet lenet)
target_link_libraries(lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(lenet_no_pool lenet_no_pool)
target_link_libraries(lenet_no_pool exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(large_image_lenet_new large_image_lenet_new)
target_link_libraries(large_image_lenet_new exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(large_image_lenet large_image_lenet)
target_link_libraries(large_image_lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)
