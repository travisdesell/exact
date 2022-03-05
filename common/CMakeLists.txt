add_executable(math_test exp)
target_compile_definitions(math_test PUBLIC -DEXACT_MATH_TEST)

add_executable(ac_test archipelago_config_test.cxx archipelago_config.cxx log.cxx arguments.cxx files.cxx)

if (MYSQL_FOUND)
    message(STATUS "mysql found, adding db_conn to exact_common library!")
    add_library(exact_common arguments.cxx random.cxx exp.cxx log.cxx db_conn.cxx color_table.cxx files.cxx weight_initialize.hxx dataset_meta.cxx archipelago_config.cxx args.cxx)
else (MYSQL_FOUND)
    add_library(exact_common arguments.cxx exp.cxx log.cxx random.cxx color_table.cxx files.cxx weight_initialize.hxx dataset_meta.cxx archipelago_config.cxx args.cxx)
endif (MYSQL_FOUND)

message("hello")
message("${CMAKE_BUILD_TYPE}")
if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(exact_common PUBLIC -DMINIMAL_LOGGING)
    message("hello")
endif (CMAKE_BUILD_TYPE MATCHES "Release")
