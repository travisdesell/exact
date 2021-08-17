if (TIFF_FOUND)
    if (MYSQL_FOUND)
        add_executable(apply_cnn_to_mosaic_db apply_cnn_to_mosaic_db.cxx)
        target_link_libraries(apply_cnn_to_mosaic_db exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)
    endif (MYSQL_FOUND)

    add_executable(apply_cnn_to_mosaic apply_cnn_to_mosaic.cxx)
    target_link_libraries(apply_cnn_to_mosaic exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

    add_executable(apply_cnn apply_cnn.cxx)
    target_link_libraries(apply_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)
endif (TIFF_FOUND)

add_executable(evaluate_large_image_cnn evaluate_large_image_cnn.cxx)
target_link_libraries(evaluate_large_image_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(evaluate_cnn evaluate_cnn.cxx)
target_link_libraries(evaluate_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(one_layer one_layer.cxx)
target_link_libraries(one_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(two_layer two_layer.cxx)
target_link_libraries(two_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(lenet lenet.cxx)
target_link_libraries(lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(lenet_no_pool lenet_no_pool.cxx)
target_link_libraries(lenet_no_pool exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(large_image_lenet_new large_image_lenet_new.cxx)
target_link_libraries(large_image_lenet_new exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(large_image_lenet large_image_lenet.cxx)
target_link_libraries(large_image_lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)
