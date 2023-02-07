#include <cstring>

#include <stdexcept>
using std::runtime_error;



#include <fstream>
using std::ifstream;
using std::istreambuf_iterator;
using std::ios;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

//for mkdir
#include <sys/stat.h>
#include <errno.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

typedef struct stat Stat;


#include "files.hxx"

string get_file_as_string(string file_path) noexcept(false) {
    //read the entire contents of the file into a string
    ifstream sites_file(file_path.c_str());

    if (!sites_file.is_open()) {
        throw runtime_error("Could not open input file '" + file_path + "'");
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


//tweaked from: https://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux/29828907
static int do_mkdir(const char *path, mode_t mode) {
    Stat            st;
    int             status = 0;

    if (stat(path, &st) != 0) {
        /* Directory does not exist. EEXIST for race condition */
        if (mkdir(path, mode) != 0 && errno != EEXIST) {
            status = -1;
        }

    } else if (!S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        status = -1;
    }

    return(status);
}

/**
 * ** mkpath - ensure all directories in path exist
 * ** Algorithm takes the pessimistic view and works top-down to ensure
 * ** each directory in path exists, rather than optimistically creating
 * ** the last element and working backwards.
 * */
int mkpath(const char *path, mode_t mode) {
    char           *pp;
    char           *sp;
    int             status;
    char           *copypath = strdup(path);

    status = 0;
    pp = copypath;
    while (status == 0 && (sp = strchr(pp, '/')) != 0) {
        //cerr << "trying to create directory: " << copypath << endl;
        if (sp != pp) {
            /* Neither root nor double slash in path */
            *sp = '\0';
            status = do_mkdir(copypath, mode);
            *sp = '/';
        }
        pp = sp + 1;
    }

    if (status == 0) {
        status = do_mkdir(path, mode);
    }

    free(copypath);
    return (status);
}


