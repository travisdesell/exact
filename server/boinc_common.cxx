#include <fstream>
using std::ifstream;
using std::istreambuf_iterator;
using std::ios;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

#include "server/boinc_common.hxx"

string get_file_as_string(string file_path) throw (int) {
    //read the entire contents of the file into a string
    ifstream sites_file(file_path.c_str());

    if (!sites_file.is_open()) {
        throw 1;
    }

    string fc;

    sites_file.seekg(0, ios::end);   
    fc.reserve(sites_file.tellg());
    sites_file.seekg(0, ios::beg);

    fc.assign((istreambuf_iterator<char>(sites_file)), istreambuf_iterator<char>());

    ostringstream oss;
    for (uint32_t i = 0; i < fc.size(); i++) {
        if (fc[i] != '\r') oss << fc[i];
    }

    return oss.str();
}


