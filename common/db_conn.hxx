#ifndef EXACT_DB_CONN_HXX
#define EXACT_DB_CONN_HXX

#include "mysql.h"

#define mysql_exact_query(query) __mysql_check(query, __FILE__, __LINE__)

extern MYSQL *exact_db_conn;

void set_db_info_filename(string _filename);

void __mysql_check(string query, const char *file, const int line);

void initialize_exact_database();

int mysql_exact_last_insert_id();

#endif
