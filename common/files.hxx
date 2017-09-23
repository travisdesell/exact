#ifndef EXACT_BOINC_COMMON_HXX
#define EXACT_BOINC_COMMON_HXX

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;


string get_file_as_string(string file_path) throw (runtime_error);

#endif
