#include <cstdio>

#include <fstream>
using std::ifstream;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;


#include "mysql.h"

#include "common/db_conn.hxx"

string db_info_filename = "../exact_mnist_batch2_db_info";

MYSQL *exact_db_conn = NULL;

void set_db_info_filename(string _filename) {
    db_info_filename = _filename;
}

void __mysql_check(string query, const char *file, const int line) {
    if (exact_db_conn == NULL) initialize_exact_database();

    mysql_query(exact_db_conn, query.c_str());

    if (mysql_errno(exact_db_conn) != 0) {
        ostringstream ex_msg;
        ex_msg << "ERROR in MySQL query: '" << query.c_str() << "'. Error: " << mysql_errno(exact_db_conn) << " -- '" << mysql_error(exact_db_conn) << "'. Thrown on " << file << ":" << line;
        fprintf(stderr, "%s\n", ex_msg.str().c_str());
        exit(1);
    }   
}

void initialize_exact_database() {
    exact_db_conn = mysql_init(NULL);

    //shoud get database info from a file
    string db_host, db_name, db_password, db_user, db_port_s;
    ifstream db_info_file(db_info_filename);

    getline(db_info_file, db_host);
    getline(db_info_file, db_name);
    getline(db_info_file, db_user);
    getline(db_info_file, db_password);
    getline(db_info_file, db_port_s);

    int db_port = stoi(db_port_s);

    db_info_file.close();

    fprintf(stderr, "parsed db info, host: '%s', name: '%s', user: '%s', pass: '%s', port: '%d'\n", db_host.c_str(), db_name.c_str(), db_user.c_str(), db_password.c_str(), db_port);

    if (mysql_real_connect(exact_db_conn, db_host.c_str(), db_user.c_str(), db_password.c_str(), db_name.c_str(), db_port, NULL, 0) == NULL) {
        fprintf(stderr, "Error connecting to database: %d, '%s'\n", mysql_errno(exact_db_conn), mysql_error(exact_db_conn));
        exit(1);
    }   
}

int mysql_exact_last_insert_id() {
    return mysql_insert_id(exact_db_conn);
}

