add_executable(math_test exp)
target_compile_definitions(math_test PUBLIC -DEXACT_MATH_TEST)

if (MYSQL_FOUND)
    message(STATUS "mysql found, adding db_conn to exact_common library!")
    add_library(exact_common arguments random exp db_conn color_table log files weight_initialize dataset_meta)
else (MYSQL_FOUND)
    add_library(exact_common arguments exp random color_table log files weight_initialize dataset_meta)
endif (MYSQL_FOUND)
