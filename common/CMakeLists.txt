add_executable(math_test exp)
target_compile_definitions(math_test PUBLIC -DEXACT_MATH_TEST)

if (MYSQL_FOUND)
    message(STATUS "mysql found, adding db_conn to exact_common library!")
    add_library(exact_common arguments.cxx random.cxx exp.cxx db_conn.cxx color_table.cxx log.cxx files.cxx)
else (MYSQL_FOUND)
    add_library(exact_common arguments.cxx exp.cxx random.cxx color_table.cxx log.cxx files.cxx)
endif (MYSQL_FOUND)
